"""Unit tests for the ContextManager."""

import json
from unittest.mock import patch

import pytest

from sia.context_manager import ContextManager


@pytest.fixture
def run_dir(tmp_path):
    """Create a temporary run directory with a minimal gen_1."""
    gen1 = tmp_path / "gen_1"
    gen1.mkdir()

    # Create a minimal target_agent.py
    (gen1 / "target_agent.py").write_text("print('hello')\n")

    return tmp_path


@pytest.fixture
def context_mgr(run_dir):
    config = {
        "task_dir": "./tasks/test-task",
        "meta_model": "haiku",
        "task_model": "haiku",
        "agent_impl": "claude",
        "max_gen": 3,
    }
    mgr = ContextManager(str(run_dir), config)
    mgr.initialize()
    return mgr


def test_initialize_creates_context_md(context_mgr, run_dir):
    ctx = run_dir / "context.md"
    assert ctx.is_file()
    content = ctx.read_text()
    assert "Run Context" in content
    assert "haiku" in content


def test_add_generation(context_mgr, run_dir):
    gen_dir = run_dir / "gen_1"

    context_mgr.add_generation(
        gen_num=1,
        gen_data={
            "success": True,
            "timestamp": "2025-01-01 00:00:00",
            "duration": 10.5,
            "agent_path": str(gen_dir / "target_agent.py"),
            "gen_dir": str(gen_dir),
            "improvement_path": None,
            "execution_type": "Single",
        },
    )

    content = (run_dir / "context.md").read_text()
    assert "Generation 1" in content
    assert "SUCCESS" in content


def test_add_generation_with_results_json(context_mgr, run_dir):
    gen_dir = run_dir / "gen_1"
    results = {"accuracy": 0.85, "n_correct": 170, "n_total": 200}
    (gen_dir / "results.json").write_text(json.dumps(results))

    context_mgr.add_generation(
        gen_num=1,
        gen_data={
            "success": True,
            "timestamp": "2025-01-01 00:00:00",
            "duration": 5.0,
            "agent_path": str(gen_dir / "target_agent.py"),
            "gen_dir": str(gen_dir),
            "improvement_path": None,
            "execution_type": "Single",
        },
    )

    content = (run_dir / "context.md").read_text()
    assert "0.85" in content


def test_finalize_with_metrics(context_mgr, run_dir):
    gen1 = run_dir / "gen_1"
    (gen1 / "results.json").write_text(json.dumps({"accuracy": 0.80}))

    context_mgr.add_generation(
        gen_num=1,
        gen_data={
            "success": True,
            "timestamp": "2025-01-01 00:00:00",
            "duration": 5.0,
            "agent_path": str(gen1 / "target_agent.py"),
            "gen_dir": str(gen1),
            "improvement_path": None,
            "execution_type": "Single",
        },
    )

    context_mgr.finalize()
    content = (run_dir / "context.md").read_text()
    assert "Summary Statistics" in content


@pytest.mark.usefixtures("run_dir")
@patch("sia.context_manager.ContextManager._generate_llm_summary", return_value=None)
def test_multiple_generations_track_deltas(mock_llm, context_mgr, run_dir):
    # Gen 1
    gen1 = run_dir / "gen_1"
    (gen1 / "results.json").write_text(json.dumps({"accuracy": 0.70}))

    context_mgr.add_generation(
        gen_num=1,
        gen_data={
            "success": True,
            "timestamp": "2025-01-01 00:00:00",
            "duration": 5.0,
            "agent_path": str(gen1 / "target_agent.py"),
            "gen_dir": str(gen1),
            "improvement_path": None,
            "execution_type": "Single",
        },
    )

    # Gen 2
    gen2 = run_dir / "gen_2"
    gen2.mkdir()
    (gen2 / "target_agent.py").write_text("print('improved')\nimport os\n")
    (gen2 / "results.json").write_text(json.dumps({"accuracy": 0.85}))
    (gen2 / "improvement.md").write_text("## Changes\n- Added better error handling\n- Improved prompt structure\n")

    context_mgr.add_generation(
        gen_num=2,
        gen_data={
            "success": True,
            "timestamp": "2025-01-01 00:01:00",
            "duration": 8.0,
            "agent_path": str(gen2 / "target_agent.py"),
            "gen_dir": str(gen2),
            "improvement_path": str(gen2 / "improvement.md"),
            "execution_type": "Single",
        },
    )

    content = (run_dir / "context.md").read_text()
    assert "Generation 2" in content
    assert "Modified by feedback agent" in content


# --- Bounded items summary (verifier->feedback contract) -------------------


def _write_results(gen_dir, payload):
    (gen_dir / "results.json").write_text(json.dumps(payload))


def test_items_summary_counts_status_group_category(context_mgr, run_dir):
    gen_dir = run_dir / "gen_1"
    _write_results(
        gen_dir,
        {
            "accuracy": 0.5,
            "items": [
                {"id": "q0", "status": "CORRECT", "group": "world_1", "category": "ok"},
                {"id": "q1", "status": "WRONG", "group": "world_1", "category": "missing_distinct"},
                {"id": "q2", "status": "WRONG", "group": "world_1", "category": "missing_distinct"},
                {"id": "q3", "status": "EXEC_ERROR", "group": "car_1", "category": "bad_join"},
            ],
        },
    )

    metrics = context_mgr._extract_metrics(str(gen_dir))

    assert metrics["accuracy"] == 0.5
    summary = metrics["items_summary"]
    assert summary["total"] == 4
    assert summary["failures"] == 3
    assert summary["status_counts"] == {"CORRECT": 1, "WRONG": 2, "EXEC_ERROR": 1}
    assert summary["group_failure_counts"] == {"world_1": 2, "car_1": 1}
    assert summary["category_counts"] == {"missing_distinct": 2, "bad_join": 1}
    assert summary["worst_ids"] == ["q1", "q2", "q3"]


def test_items_summary_derives_category_when_absent(context_mgr, run_dir):
    gen_dir = run_dir / "gen_1"
    _write_results(
        gen_dir,
        {
            "items": [
                {"id": "q0", "status": "CORRECT", "group": "g1"},
                {"id": "q1", "status": "WRONG", "group": "g1"},
            ],
        },
    )

    summary = context_mgr._extract_metrics(str(gen_dir))["items_summary"]

    assert summary["failures"] == 1
    assert summary["group_failure_counts"] == {"g1": 1}
    # No category on items -> coarse category derived from status.
    assert summary["category_counts"] == {"status:WRONG": 1}


def test_no_items_key_produces_no_summary(context_mgr, run_dir):
    gen_dir = run_dir / "gen_1"
    _write_results(gen_dir, {"accuracy": 0.9, "correct": 9, "total": 10})

    metrics = context_mgr._extract_metrics(str(gen_dir))

    assert "items_summary" not in metrics
    assert metrics["accuracy"] == 0.9
    assert metrics["correct"] == 9


def test_items_summary_stays_bounded_on_large_array(context_mgr, run_dir):
    from sia.context_manager import (
        ITEMS_SUMMARY_MAX_CATEGORIES,
        ITEMS_SUMMARY_MAX_GROUPS,
        ITEMS_SUMMARY_MAX_WORST_IDS,
    )

    gen_dir = run_dir / "gen_1"
    items = [{"id": f"q{i}", "status": "WRONG", "group": f"g{i}", "category": f"c{i}"} for i in range(10_000)]
    _write_results(gen_dir, {"items": items})

    summary = context_mgr._extract_metrics(str(gen_dir))["items_summary"]

    assert summary["total"] == 10_000
    assert summary["failures"] == 10_000
    assert len(summary["group_failure_counts"]) <= ITEMS_SUMMARY_MAX_GROUPS
    assert len(summary["category_counts"]) <= ITEMS_SUMMARY_MAX_CATEGORIES
    assert len(summary["worst_ids"]) <= ITEMS_SUMMARY_MAX_WORST_IDS
