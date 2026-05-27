"""Unit tests for the ContextManager."""

import json

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
        "backend": "claude",
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


def test_multiple_generations_track_deltas(context_mgr, run_dir):
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
