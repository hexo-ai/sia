"""Temporal experiment knowledge graph for SIA runs.

The graph is intentionally lightweight: append-only observations stored as JSON.
It mirrors the useful parts of an RDF-style temporal graph without introducing a
database dependency into SIA's default path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sia.io_utils import safe_load_json, safe_read_file
from sia.layout import Names

SCHEMA_VERSION = 1
KNOWLEDGE_GRAPH_JSON = "knowledge_graph.json"
KNOWLEDGE_GRAPH_MD = "knowledge_graph.md"
METRIC_RESULT_FILES = (
    Names.RESULTS_JSON,
    "evaluation_results.json",
    "detailed_results.json",
)

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
_SCALAR_TYPES = (str, int, float, bool)
_PRIMARY_METRIC_HINTS = ("accuracy", "score", "pass_rate", "pass@1", "f1", "auc", "mse", "loss")


@dataclass
class GraphObservation:
    """One timestamped subject-predicate-object observation."""

    subject: str
    predicate: str
    object: str
    observed_at: str
    source: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    """Append-only run knowledge graph."""

    schema_version: int = SCHEMA_VERSION
    observations: list[GraphObservation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def utc_now_iso() -> str:
    """Return a compact UTC timestamp suitable for graph observations."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_graph(task_dir: str, run_dir: str) -> KnowledgeGraph:
    """Create an initialized graph for a run."""

    graph = KnowledgeGraph(
        metadata={
            "task_dir": task_dir,
            "run_dir": run_dir,
            "created_at": utc_now_iso(),
        }
    )
    task_id = _task_id(task_dir)
    add_observation(
        graph,
        subject=task_id,
        predicate="instance_of",
        object="type:task",
        source=task_dir,
    )
    add_observation(
        graph,
        subject=f"run:{Path(run_dir).name}",
        predicate="evaluates",
        object=task_id,
        source=run_dir,
    )
    return graph


def load_graph(path: str | Path) -> KnowledgeGraph:
    """Load a graph from JSON."""

    raw = safe_load_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Knowledge graph must be a JSON object: {path}")
    observations = [
        GraphObservation(
            subject=str(item["subject"]),
            predicate=str(item["predicate"]),
            object=str(item["object"]),
            observed_at=str(item["observed_at"]),
            source=item.get("source"),
            properties=item.get("properties") if isinstance(item.get("properties"), dict) else {},
        )
        for item in raw.get("observations", [])
        if isinstance(item, dict)
        and all(key in item for key in ("subject", "predicate", "object", "observed_at"))
    ]
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    return KnowledgeGraph(
        schema_version=int(raw.get("schema_version", SCHEMA_VERSION)),
        observations=observations,
        metadata=metadata,
    )


def save_graph(graph: KnowledgeGraph, path: str | Path) -> None:
    """Persist a graph as stable, readable JSON."""

    payload = {
        "schema_version": graph.schema_version,
        "metadata": graph.metadata,
        "observations": [
            {
                "subject": obs.subject,
                "predicate": obs.predicate,
                "object": obs.object,
                "observed_at": obs.observed_at,
                "source": obs.source,
                "properties": obs.properties,
            }
            for obs in graph.observations
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def add_observation(
    graph: KnowledgeGraph,
    *,
    subject: str,
    predicate: str,
    object: str,
    source: str | None = None,
    properties: dict[str, Any] | None = None,
    observed_at: str | None = None,
) -> None:
    """Append one graph observation.

    Identical triples are deliberately not deduped. Repeated sightings are useful
    temporal evidence across generations.
    """

    graph.observations.append(
        GraphObservation(
            subject=subject,
            predicate=predicate,
            object=object,
            observed_at=observed_at or utc_now_iso(),
            source=source,
            properties=properties or {},
        )
    )


def graph_json_path(run_dir: str | Path) -> str:
    """Path to the run's machine-readable knowledge graph."""

    return str(Path(run_dir) / KNOWLEDGE_GRAPH_JSON)


def graph_markdown_path(run_dir: str | Path) -> str:
    """Path to the run's human-readable knowledge graph digest."""

    return str(Path(run_dir) / KNOWLEDGE_GRAPH_MD)


def extract_metrics(gen_dir: str | Path) -> dict[str, Any]:
    """Extract top-level scalar metrics from a generation's results file."""

    for filename in METRIC_RESULT_FILES:
        path = Path(gen_dir) / filename
        if not path.exists():
            continue
        data = safe_load_json(path)
        if isinstance(data, dict):
            metrics = {str(k): v for k, v in data.items() if isinstance(v, _SCALAR_TYPES)}
            if metrics:
                return metrics
    return {}


def extract_improvement_sections(improvement_path: str | Path) -> dict[str, str]:
    """Extract useful sections from an improvement markdown file.

    The feedback prompt will ask for stable headings, but existing runs may only
    have bullets. Missing headings therefore fall back to a short summary block.
    """

    content = safe_read_file(improvement_path)
    if not content:
        return {}

    sections: dict[str, list[str]] = {}
    current_key: str | None = None
    preface: list[str] = []

    for line in content.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            current_key = _normalize_section_key(match.group(1))
            sections.setdefault(current_key, [])
            continue
        if current_key:
            sections[current_key].append(line)
        else:
            preface.append(line)

    cleaned = {key: _clean_section(lines) for key, lines in sections.items()}
    cleaned = {key: value for key, value in cleaned.items() if value}
    if cleaned:
        return cleaned

    bullets = _extract_bullets(content)
    if bullets:
        return {"summary": "\n".join(bullets[:5])}
    summary = _first_nonempty_lines(content, limit=5)
    return {"summary": summary} if summary else {}


def classify_outcome(previous_metrics: dict[str, Any], current_metrics: dict[str, Any]) -> str:
    """Classify the generation outcome from metric movement."""

    metric_name = _select_primary_metric(previous_metrics, current_metrics)
    if not metric_name:
        return "inconclusive"

    previous = _coerce_number(previous_metrics.get(metric_name))
    current = _coerce_number(current_metrics.get(metric_name))
    if previous is None or current is None:
        return "inconclusive"

    lower_is_better = metric_name.lower() in {"loss", "mse", "rmse", "mae", "error"}
    if current == previous:
        return "inconclusive"
    improved = current < previous if lower_is_better else current > previous
    return "supported" if improved else "contradicted"


def update_after_generation(
    graph: KnowledgeGraph,
    gen_num: int,
    gen_dir: str | Path,
    prev_gen_dir: str | Path | None = None,
) -> None:
    """Append observations extracted from one generation's artifacts."""

    gen_path = Path(gen_dir)
    gen_id = f"generation:{gen_num}"
    source = str(gen_path)
    add_observation(
        graph,
        subject=gen_id,
        predicate="instance_of",
        object="type:generation",
        source=source,
        properties={"generation": gen_num},
    )

    _add_artifact_if_present(graph, gen_id, gen_path / Names.TARGET_AGENT, "artifact:target_agent")
    _add_artifact_if_present(graph, gen_id, gen_path / Names.TRAIN_SCRIPT, "artifact:train_script")
    _add_artifact_if_present(graph, gen_id, gen_path / Names.IMPROVEMENT_MD, "artifact:improvement_plan")

    metrics = extract_metrics(gen_path)
    result_source = _metric_source(gen_path)
    for name, value in metrics.items():
        metric_id = f"metric:gen_{gen_num}:{_slug(name)}"
        add_observation(
            graph,
            subject=gen_id,
            predicate="has_metric",
            object=metric_id,
            source=str(result_source) if result_source else None,
            properties={"name": name, "value": value},
        )

    improvement_path = gen_path / Names.IMPROVEMENT_MD
    if improvement_path.exists():
        sections = extract_improvement_sections(improvement_path)
        experiment_id = f"experiment:gen_{gen_num}"
        add_observation(
            graph,
            subject=experiment_id,
            predicate="observed_in",
            object=gen_id,
            source=str(improvement_path),
            properties={"generation": gen_num},
        )
        _add_improvement_observations(graph, experiment_id, sections, str(improvement_path))

        previous_metrics = extract_metrics(prev_gen_dir) if prev_gen_dir else {}
        outcome = classify_outcome(previous_metrics, metrics)
        add_observation(
            graph,
            subject=experiment_id,
            predicate=outcome,
            object=gen_id,
            source=str(result_source) if result_source else None,
            properties={"previous_metrics": previous_metrics, "current_metrics": metrics},
        )

    _add_failure_observations(graph, gen_id, gen_path)


def render_digest(graph: KnowledgeGraph, max_items: int = 12) -> str:
    """Render a compact graph digest for prompt injection."""

    if not graph.observations:
        return "No experiment knowledge graph observations yet."

    lines = ["Recent experiment knowledge:"]
    recent = graph.observations[-max_items:]
    for obs in recent:
        detail = _observation_detail(obs)
        lines.append(f"- {obs.subject} {obs.predicate} {obs.object}{detail}")

    lessons = _objects_for_predicate(graph, "produced")
    failures = _objects_for_predicate(graph, "observed_in", prefix="failure:")
    if lessons:
        lines.append("")
        lines.append("Reusable lessons:")
        lines.extend(f"- {item}" for item in lessons[-5:])
    if failures:
        lines.append("")
        lines.append("Open failure modes:")
        lines.extend(f"- {item}" for item in failures[-5:])
    return "\n".join(lines)


def write_markdown_digest(graph: KnowledgeGraph, path: str | Path) -> None:
    """Write a human-readable graph digest for the run directory."""

    lines = [
        "# Experiment Knowledge Graph",
        "",
        f"Schema version: {graph.schema_version}",
        f"Observations: {len(graph.observations)}",
        "",
        "## Digest",
        "",
        render_digest(graph, max_items=25),
        "",
        "## Observations",
        "",
    ]
    for obs in graph.observations:
        detail = _observation_detail(obs)
        source = f" source={obs.source}" if obs.source else ""
        lines.append(f"- `{obs.observed_at}` {obs.subject} {obs.predicate} {obs.object}{detail}{source}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _add_artifact_if_present(graph: KnowledgeGraph, gen_id: str, artifact_path: Path, artifact_type: str) -> None:
    if not artifact_path.exists():
        return
    artifact_id = f"artifact:{artifact_path.parent.name}:{artifact_path.name}"
    add_observation(
        graph,
        subject=gen_id,
        predicate="produced",
        object=artifact_id,
        source=str(artifact_path),
        properties={"artifact_type": artifact_type, "path": str(artifact_path)},
    )


def _metric_source(gen_path: Path) -> Path | None:
    for filename in METRIC_RESULT_FILES:
        path = gen_path / filename
        if path.exists():
            return path
    return None


def _add_improvement_observations(
    graph: KnowledgeGraph,
    experiment_id: str,
    sections: dict[str, str],
    source: str,
) -> None:
    field_map = {
        "hypothesis": "tested",
        "planned_change": "changed",
        "change": "changed",
        "evidence": "derived_from",
        "expected_impact": "expects",
        "risk": "risks",
        "summary": "describes",
    }
    for key, predicate in field_map.items():
        value = sections.get(key)
        if not value:
            continue
        object_id = f"{key}:{_slug(value)[:80]}"
        add_observation(
            graph,
            subject=experiment_id,
            predicate=predicate,
            object=object_id,
            source=source,
            properties={"text": value},
        )


def _add_failure_observations(graph: KnowledgeGraph, gen_id: str, gen_path: Path) -> None:
    stdout_paths = [gen_path / Names.STDOUT_LOG, gen_path / Names.TRAIN_STDOUT_LOG, gen_path / Names.EVAL_LOG]
    for path in stdout_paths:
        content = safe_read_file(path) if path.exists() else None
        if not content:
            continue
        lower = content.lower()
        if "traceback" in lower or "error" in lower or "failed" in lower:
            failure = "failure:execution_error"
            if "timeout" in lower:
                failure = "failure:timeout"
            elif "json" in lower:
                failure = "failure:json_or_logging_error"
            add_observation(graph, subject=failure, predicate="observed_in", object=gen_id, source=str(path))
            return


def _select_primary_metric(previous_metrics: dict[str, Any], current_metrics: dict[str, Any]) -> str | None:
    keys = list(current_metrics) or list(previous_metrics)
    lower_to_key = {key.lower(): key for key in keys}
    for hint in _PRIMARY_METRIC_HINTS:
        if hint in lower_to_key:
            return lower_to_key[hint]
    for key in keys:
        if _coerce_number(current_metrics.get(key, previous_metrics.get(key))) is not None:
            return key
    return None


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        text = value.strip().removesuffix("%")
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _normalize_section_key(raw: str) -> str:
    text = raw.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    aliases = {
        "planned_changes": "planned_change",
        "plan": "planned_change",
        "change": "planned_change",
        "changes": "planned_change",
        "expected_metric_impact": "expected_impact",
        "impact": "expected_impact",
        "risks": "risk",
    }
    return aliases.get(text, text)


def _clean_section(lines: list[str]) -> str:
    return "\n".join(line.strip() for line in lines).strip()


def _extract_bullets(content: str) -> list[str]:
    bullets = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- ", "* ")):
            bullets.append(stripped[2:].strip())
        elif re.match(r"^\d+\.\s+", stripped):
            bullets.append(re.sub(r"^\d+\.\s+", "", stripped).strip())
    return [item for item in bullets if item]


def _first_nonempty_lines(content: str, limit: int) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    return "\n".join(lines[:limit])


def _observation_detail(obs: GraphObservation) -> str:
    text = obs.properties.get("text")
    if isinstance(text, str) and text:
        return f" ({_one_line(text)})"
    if "value" in obs.properties:
        name = obs.properties.get("name", "value")
        return f" ({name}={obs.properties['value']})"
    return ""


def _objects_for_predicate(graph: KnowledgeGraph, predicate: str, prefix: str | None = None) -> list[str]:
    values = []
    for obs in graph.observations:
        if obs.predicate != predicate:
            continue
        value = obs.object
        if prefix and not value.startswith(prefix):
            continue
        values.append(value)
    return values


def _task_id(task_dir: str) -> str:
    name = Path(task_dir).name or "task"
    return f"task:{_slug(name)}"


def _slug(value: Any) -> str:
    text = _one_line(str(value)).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def _one_line(value: str, limit: int = 160) -> str:
    text = " ".join(value.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."
