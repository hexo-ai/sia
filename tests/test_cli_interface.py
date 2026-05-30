"""Verify CLI interface is unchanged after refactoring."""

import subprocess
import sys


def test_help_exits_zero():
    """sia --help should exit with code 0."""
    result = subprocess.run(
        [sys.executable, "-m", "sia", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--max_gen" in result.stdout
    assert "--task" in result.stdout
    assert "--task_dir" in result.stdout
    assert "--meta_model" in result.stdout
    assert "--task_model" in result.stdout
    assert "--backend" in result.stdout
    assert "--sandbox" in result.stdout


def test_no_args_exits_nonzero():
    """sia without required args should exit non-zero."""
    result = subprocess.run(
        [sys.executable, "-m", "sia"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_invalid_task_exits_nonzero():
    """sia --task nonexistent should exit non-zero."""
    result = subprocess.run(
        [sys.executable, "-m", "sia", "--task", "nonexistent"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
