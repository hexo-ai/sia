"""Integration tests for generation loop with mocked agents."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from sia.config import Config
from sia.context_manager import ContextManager
from sia.orchestrator import (
    RunSetup,
    TaskFiles,
    _run_target_agent,
    run_generation,
)


def _make_task_files(tmp_path):
    """Create minimal task structure with all required files."""
    task_dir = tmp_path / "task"
    shared_dir = task_dir / "_shared"
    ref_dir = task_dir / "reference"
    pub_dir = task_dir / "data" / "public"

    for d in (shared_dir, ref_dir, pub_dir):
        d.mkdir(parents=True)

    (ref_dir / "SAMPLE_TASK_DESCRIPTIONS.md").write_text("Sample task description text.")
    (ref_dir / "reference_target_agent.py").write_text("print('ref agent')")
    (shared_dir / "sample_agent_execution.json").write_text(json.dumps([{"role": "user"}]))
    (pub_dir / "task.md").write_text("# Test task\nSolve the problem.")
    return task_dir, shared_dir


def _make_run_setup(tmp_path, task_dir):
    """Create a RunSetup with initialized context manager."""
    run_dir = tmp_path / "runs" / "run_1"
    gen1 = run_dir / "gen_1"
    gen1.mkdir(parents=True)
    (gen1 / "target_agent.py").write_text("print('agent')\n")

    context_mgr = ContextManager(
        str(run_dir),
        {
            "task_dir": str(task_dir),
            "meta_model": "haiku",
            "task_model": "haiku",
            "backend": "claude",
            "max_gen": 1,
        },
    )
    context_mgr.initialize()

    return RunSetup(
        run_directory=str(run_dir),
        meta_agent_working_directory=str(gen1),
        venv_dir=str(tmp_path / "venv"),
        context_mgr=context_mgr,
    )


@patch("sia.orchestrator.subprocess.Popen")
def test_run_target_agent_success(mock_popen_cls, tmp_path):
    """_run_target_agent with sandbox=none uses standard Popen path."""
    gen_dir = tmp_path / "gen_1"
    gen_dir.mkdir()
    stdout_log = str(gen_dir / "stdout.log")
    (gen_dir / "target_agent.py").write_text("print('ok')")

    # Mock Popen to simulate a process that writes one line then exits 0
    mock_process = MagicMock()
    mock_process.stdout = iter(["line1\n"])
    mock_process.wait.return_value = 0
    mock_popen_cls.return_value = mock_process

    success, _stdout, _stderr, err = _run_target_agent(
        venv_dir="/fake/venv",
        target_agent_path=str(gen_dir / "target_agent.py"),
        abs_dataset_dir="/data",
        gen_dir=str(gen_dir),
        stdout_log_file=stdout_log,
        sandbox="none",
        env_config=Config(),
    )

    assert success is True
    assert err == ""
    mock_popen_cls.assert_called_once()
    # Verify no Docker args in the command
    cmd = mock_popen_cls.call_args[0][0]
    assert "docker" not in cmd


@patch("sia.orchestrator.subprocess.Popen")
def test_run_target_agent_failure(mock_popen_cls, tmp_path):
    """_run_target_agent returns (False, ...) on non-zero exit."""
    gen_dir = tmp_path / "gen_1"
    gen_dir.mkdir()
    stdout_log = str(gen_dir / "stdout.log")
    (gen_dir / "target_agent.py").write_text("raise SystemExit(1)")

    mock_process = MagicMock()
    mock_process.stdout = iter(["error\n"])
    mock_process.wait.return_value = 1
    mock_popen_cls.return_value = mock_process

    success, _stdout, _stderr, err = _run_target_agent(
        venv_dir="/fake/venv",
        target_agent_path=str(gen_dir / "target_agent.py"),
        abs_dataset_dir="/data",
        gen_dir=str(gen_dir),
        stdout_log_file=stdout_log,
        sandbox="none",
        env_config=Config(),
    )

    assert success is False
    assert "exit code 1" in err


@patch("sia.orchestrator._run_feedback_agent")
@patch("sia.orchestrator._run_target_agent")
def test_single_generation_creates_context(mock_run_ta, mock_run_fb, tmp_path):
    """run_generation with max_gen=1 creates context.md entry."""
    task_dir, _shared_dir = _make_task_files(tmp_path)
    run_setup = _make_run_setup(tmp_path, task_dir)

    mock_run_ta.return_value = (True, "output", "", "")

    task_files = TaskFiles(
        sample_task_descriptions="desc",
        reference_target_agent_py="ref",
        sample_agent_execution={},
        task_md="# Task",
    )

    run_generation(
        current_gen=1,
        max_gen=1,
        run_setup=run_setup,
        task_files=task_files,
        abs_dataset_dir=str(task_dir / "data" / "public"),
        dataset_dir=str(task_dir / "data" / "public"),
        meta_model="haiku",
        backend="claude",
        sandbox="none",
        env_config=Config(),
    )

    # Verify context.md was updated
    ctx = (Path(run_setup.run_directory) / "context.md").read_text()
    assert "Generation 1" in ctx
    assert "SUCCESS" in ctx

    # Feedback agent should NOT be called (last generation)
    mock_run_fb.assert_not_called()


@patch("sia.orchestrator._run_feedback_agent")
@patch("sia.orchestrator._run_target_agent")
def test_run_generation_directory_structure(mock_run_ta, mock_run_fb, tmp_path):
    """Verify gen directory structure is preserved after run."""
    task_dir, _ = _make_task_files(tmp_path)
    run_setup = _make_run_setup(tmp_path, task_dir)

    mock_run_ta.return_value = (True, "output", "", "")

    run_generation(
        current_gen=1,
        max_gen=1,
        run_setup=run_setup,
        task_files=TaskFiles("d", "r", {}, "# T"),
        abs_dataset_dir="/data",
        dataset_dir="/data",
        meta_model="haiku",
        backend="claude",
        sandbox="none",
        env_config=Config(),
    )

    gen_dir = Path(run_setup.run_directory) / "gen_1"
    assert gen_dir.is_dir()
    assert (gen_dir / "target_agent.py").is_file()


@patch("sia.context_manager.ContextManager._generate_llm_summary", return_value=None)
@patch("sia.orchestrator._run_feedback_agent")
@patch("sia.orchestrator._run_target_agent")
def test_two_generations_with_feedback(mock_run_ta, mock_run_fb, mock_llm, tmp_path):
    """Two-generation evolution: feedback agent called for gen_1, skipped for gen_2."""
    task_dir, _ = _make_task_files(tmp_path)
    run_setup = _make_run_setup(tmp_path, task_dir)

    mock_run_ta.return_value = (True, "output", "", "")

    # Stub _run_feedback_agent to create gen_2/target_agent.py
    def _fake_feedback(*args, **kwargs):
        next_gen_dir = Path(run_setup.run_directory) / "gen_2"
        next_gen_dir.mkdir(exist_ok=True)
        (next_gen_dir / "target_agent.py").write_text("print('improved')\n")
        (next_gen_dir / "improvement.md").write_text("- Better prompts\n- More robust error handling\n")

    mock_run_fb.side_effect = _fake_feedback

    task_files = TaskFiles("d", "r", {}, "# T")

    # Generation 1 (should trigger feedback agent)
    run_generation(
        current_gen=1,
        max_gen=2,
        run_setup=run_setup,
        task_files=task_files,
        abs_dataset_dir="/data",
        dataset_dir="/data",
        meta_model="haiku",
        backend="claude",
        sandbox="none",
        env_config=Config(),
    )
    mock_run_fb.assert_called_once()

    # Generation 2 (should NOT trigger feedback agent -- last generation)
    run_generation(
        current_gen=2,
        max_gen=2,
        run_setup=run_setup,
        task_files=task_files,
        abs_dataset_dir="/data",
        dataset_dir="/data",
        meta_model="haiku",
        backend="claude",
        sandbox="none",
        env_config=Config(),
    )
    assert mock_run_fb.call_count == 1  # still only called once

    # Verify both gen directories exist
    run_dir = Path(run_setup.run_directory)
    assert (run_dir / "gen_1" / "target_agent.py").is_file()
    assert (run_dir / "gen_2" / "target_agent.py").is_file()

    # Verify context.md tracks both generations
    ctx = (run_dir / "context.md").read_text()
    assert "Generation 1" in ctx
    assert "Generation 2" in ctx

    # Verify finalize produces summary
    run_setup.context_mgr.finalize()
    ctx_final = (run_dir / "context.md").read_text()
    assert "Summary Statistics" in ctx_final
    assert "**Total Generations**: 2" in ctx_final
