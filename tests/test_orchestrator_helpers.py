"""Unit tests for orchestrator helper functions."""

import json
import math

from sia.orchestrator import (
    _build_transfer_evidence_card,
    load_agent_execution,
)


def test_load_single_trajectory(tmp_path):
    trajectory = [{"role": "user", "content": "hello"}]
    (tmp_path / "agent_execution.json").write_text(json.dumps(trajectory))

    data, is_multi = load_agent_execution(str(tmp_path))
    assert not is_multi
    assert isinstance(data, list)
    assert data[0]["role"] == "user"


def test_load_multi_trajectory(tmp_path):
    exec_dir = tmp_path / "agent_execution"
    exec_dir.mkdir()

    for i in range(3):
        traj = [{"role": "user", "content": f"question {i}"}]
        (exec_dir / f"execution_q{i}.json").write_text(json.dumps(traj))

    data, is_multi = load_agent_execution(str(tmp_path))
    assert is_multi
    assert data["count"] == 3
    assert len(data["trajectories"]) == 3


def test_load_missing_execution(tmp_path):
    data, _is_multi = load_agent_execution(str(tmp_path))
    assert "error" in data


def test_load_malformed_json(tmp_path):
    (tmp_path / "agent_execution.json").write_text("{not valid json")

    data, is_multi = load_agent_execution(str(tmp_path))
    assert not is_multi
    assert "error" in data or "raw_preview" in data


def test_load_empty_multi_trajectory_folder(tmp_path):
    (tmp_path / "agent_execution").mkdir()

    data, is_multi = load_agent_execution(str(tmp_path))
    assert is_multi
    assert "error" in data


def test_build_transfer_evidence_card_keeps_reusable_and_residue(tmp_path):
    gen1 = tmp_path / "gen_1"
    gen2 = tmp_path / "gen_2"
    gen1.mkdir()
    gen2.mkdir()

    (gen1 / "results.json").write_text(json.dumps({"accuracy": 0.7}))
    (gen2 / "results.json").write_text(json.dumps({"accuracy": 0.85}))
    (gen2 / "improvement.md").write_text(
        "\n".join(
            [
                "# Improvement Plan",
                "- Added generic prompt retry flow.",
                "- This task-specific branch added a hardcoded guard for sample 17.",
            ]
        )
    )

    card = _build_transfer_evidence_card(
        current_gen=2,
        gen_dir=str(gen2),
        improvement_path=str(gen2 / "improvement.md"),
        evaluation_result={"status": "success"},
    )

    assert card.score_key == "accuracy"
    assert math.isclose(card.score_delta, 0.15, rel_tol=0, abs_tol=1e-12)
    assert card.reusable_bullets == ["Added generic prompt retry flow."]
    assert card.residue_bullets == ["This task-specific branch added a hardcoded guard for sample 17."]


def test_build_transfer_evidence_card_marks_missing_data(tmp_path):
    gen1 = tmp_path / "gen_1"
    gen1.mkdir()
    (gen1 / "results.json").write_text(json.dumps({"notes": "skipped"}))

    card = _build_transfer_evidence_card(
        current_gen=1,
        gen_dir=str(gen1),
        improvement_path=None,
        evaluation_result={"status": "warning"},
    )

    assert card.evaluator_status == "evaluator_results_missing"
    assert card.score_key is None
    assert card.unsupported_claims
