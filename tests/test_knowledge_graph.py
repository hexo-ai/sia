import json

from sia.knowledge_graph import (
    add_observation,
    classify_outcome,
    extract_improvement_sections,
    extract_metrics,
    load_graph,
    new_graph,
    render_digest,
    save_graph,
    update_after_generation,
)


def test_save_load_preserves_temporal_duplicates(tmp_path):
    graph = new_graph(str(tmp_path / "task"), str(tmp_path / "run_1"))
    add_observation(
        graph,
        subject="experiment:gen_2",
        predicate="addresses",
        object="failure:malformed_outputs",
        observed_at="2026-06-06T10:00:00Z",
    )
    add_observation(
        graph,
        subject="experiment:gen_2",
        predicate="addresses",
        object="failure:malformed_outputs",
        observed_at="2026-06-06T10:01:00Z",
    )

    path = tmp_path / "knowledge_graph.json"
    save_graph(graph, path)
    loaded = load_graph(path)

    duplicate_observations = [
        obs
        for obs in loaded.observations
        if obs.subject == "experiment:gen_2"
        and obs.predicate == "addresses"
        and obs.object == "failure:malformed_outputs"
    ]
    assert len(duplicate_observations) == 2
    assert duplicate_observations[0].observed_at != duplicate_observations[1].observed_at


def test_extract_metrics_reads_top_level_scalars(tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps(
            {
                "accuracy": 0.75,
                "correct": 15,
                "total": 20,
                "per_class": {"a": 1},
                "samples": [1, 2],
            }
        ),
        encoding="utf-8",
    )

    assert extract_metrics(tmp_path) == {"accuracy": 0.75, "correct": 15, "total": 20}


def test_extract_metrics_falls_back_to_evaluation_results_json(tmp_path):
    (tmp_path / "evaluation_results.json").write_text(
        json.dumps({"accuracy": 1.0, "correct": 3, "missing": 195, "details": [{"status": "missing"}]}),
        encoding="utf-8",
    )

    assert extract_metrics(tmp_path) == {"accuracy": 1.0, "correct": 3, "missing": 195}


def test_extract_metrics_prefers_results_json(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({"accuracy": 0.25}), encoding="utf-8")
    (tmp_path / "evaluation_results.json").write_text(json.dumps({"accuracy": 1.0}), encoding="utf-8")

    assert extract_metrics(tmp_path) == {"accuracy": 0.25}


def test_extract_improvement_sections_from_headings(tmp_path):
    path = tmp_path / "improvement.md"
    path.write_text(
        """# Hypothesis
Output validation will reduce malformed submissions.

## Evidence
- Gen 1 emitted invalid JSON.

## Planned Change
Add schema validation and one retry.
""",
        encoding="utf-8",
    )

    sections = extract_improvement_sections(path)

    assert "Output validation" in sections["hypothesis"]
    assert "invalid JSON" in sections["evidence"]
    assert "schema validation" in sections["planned_change"]


def test_extract_improvement_sections_falls_back_to_bullets(tmp_path):
    path = tmp_path / "improvement.md"
    path.write_text(
        """- Add better output parsing.
- Retry malformed responses.
""",
        encoding="utf-8",
    )

    sections = extract_improvement_sections(path)

    assert sections == {"summary": "Add better output parsing.\nRetry malformed responses."}


def test_classify_outcome_uses_primary_metric_direction():
    assert classify_outcome({"accuracy": 0.4}, {"accuracy": 0.5}) == "supported"
    assert classify_outcome({"accuracy": 0.5}, {"accuracy": 0.4}) == "contradicted"
    assert classify_outcome({"loss": 0.5}, {"loss": 0.4}) == "supported"
    assert classify_outcome({}, {"accuracy": 0.5}) == "inconclusive"


def test_update_after_generation_adds_experiment_and_metric_observations(tmp_path):
    gen1 = tmp_path / "gen_1"
    gen2 = tmp_path / "gen_2"
    gen1.mkdir()
    gen2.mkdir()
    (gen1 / "results.json").write_text(json.dumps({"accuracy": 0.4}), encoding="utf-8")
    (gen2 / "results.json").write_text(json.dumps({"accuracy": 0.5}), encoding="utf-8")
    (gen2 / "target_agent.py").write_text("print('agent')\n", encoding="utf-8")
    (gen2 / "improvement.md").write_text(
        """## Hypothesis
Validation improves answer formatting.

## Planned Change
Add JSON schema validation.
""",
        encoding="utf-8",
    )

    graph = new_graph(str(tmp_path / "task"), str(tmp_path / "run_1"))
    update_after_generation(graph, 2, gen2, prev_gen_dir=gen1)

    triples = {(obs.subject, obs.predicate, obs.object) for obs in graph.observations}
    assert ("generation:2", "has_metric", "metric:gen_2:accuracy") in triples
    assert ("experiment:gen_2", "supported", "generation:2") in triples
    assert any(obs.predicate == "tested" and "Validation improves" in obs.properties.get("text", "") for obs in graph.observations)


def test_render_digest_includes_recent_observations_and_details(tmp_path):
    graph = new_graph(str(tmp_path / "task"), str(tmp_path / "run_1"))
    add_observation(
        graph,
        subject="experiment:gen_2",
        predicate="tested",
        object="hypothesis:validation",
        properties={"text": "Validation improves answer formatting."},
    )

    digest = render_digest(graph)

    assert "Recent experiment knowledge" in digest
    assert "experiment:gen_2 tested hypothesis:validation" in digest
    assert "Validation improves answer formatting" in digest
