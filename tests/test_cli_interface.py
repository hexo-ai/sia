"""Verify CLI interface: ``run`` / ``web`` sub-commands and backward-compatible default."""

import subprocess
import sys


def _sia(*args):
    return subprocess.run([sys.executable, "-m", "sia", *args], capture_output=True, text=True)


def test_top_level_help_lists_subcommands():
    """sia --help should exit 0 and advertise both sub-commands."""
    result = _sia("--help")
    assert result.returncode == 0
    assert "run" in result.stdout
    assert "web" in result.stdout


def test_run_help_exposes_orchestrator_flags():
    """sia run --help should show the orchestrator flags."""
    result = _sia("run", "--help")
    assert result.returncode == 0
    for flag in (
        "--max_gen",
        "--task",
        "--task_dir",
        "--meta-agent-profile",
        "--target-agent-profile",
        "--sandbox",
        "--no-knowledge-graph",
    ):
        assert flag in result.stdout


def test_web_help_exposes_server_flags():
    """sia web --help should show the server flags."""
    result = _sia("web", "--help")
    assert result.returncode == 0
    for flag in ("--host", "--port", "--runs-dir"):
        assert flag in result.stdout


def test_no_args_exits_nonzero():
    """sia without a task (defaults to `run`, which requires one) exits non-zero."""
    assert _sia().returncode != 0


def test_default_subcommand_is_run():
    """`sia --task nonexistent` is treated as `sia run --task nonexistent`."""
    assert _sia("--task", "nonexistent").returncode != 0


def test_invalid_task_exits_nonzero():
    result = _sia("run", "--task", "nonexistent")
    assert result.returncode != 0
