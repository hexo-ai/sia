"""Tests for orchestrator.run_evaluation with mocked subprocess."""

import json
from unittest.mock import MagicMock, patch

from sia.orchestrator import run_evaluation


def test_skipped_when_no_evaluate_py(tmp_path):
    gen_dir = tmp_path / "gen_1"
    gen_dir.mkdir()
    task_dir = tmp_path / "task"
    task_dir.mkdir()

    result = run_evaluation(str(gen_dir), str(task_dir), "/fake/venv")
    assert result["status"] == "skipped"


def _make_task_with_eval(task_dir):
    """Create a minimal task dir with evaluate.py in data/public/."""
    pub = task_dir / "data" / "public"
    pub.mkdir(parents=True)
    (pub / "evaluate.py").write_text("pass")


@patch("sia.orchestrator.subprocess.run")
def test_success_when_results_json_created(mock_run, tmp_path):
    gen_dir = tmp_path / "gen_1"
    gen_dir.mkdir()
    (gen_dir / "results.json").write_text(json.dumps({"accuracy": 0.9}))
    _make_task_with_eval(tmp_path / "task")

    mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

    result = run_evaluation(str(gen_dir), str(tmp_path / "task"), "/fake/venv")
    assert result["status"] == "success"
    mock_run.assert_called_once()


@patch("sia.orchestrator.subprocess.run")
def test_error_on_nonzero_exit(mock_run, tmp_path):
    gen_dir = tmp_path / "gen_1"
    gen_dir.mkdir()
    _make_task_with_eval(tmp_path / "task")

    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="traceback")

    result = run_evaluation(str(gen_dir), str(tmp_path / "task"), "/fake/venv")
    assert result["status"] == "error"
    assert "code 1" in result["reason"]


@patch("sia.orchestrator.subprocess.run", side_effect=__import__("subprocess").TimeoutExpired(cmd="eval", timeout=600))
def test_timeout_handled(mock_run, tmp_path):
    gen_dir = tmp_path / "gen_1"
    gen_dir.mkdir()
    _make_task_with_eval(tmp_path / "task")

    result = run_evaluation(str(gen_dir), str(tmp_path / "task"), "/fake/venv")
    assert result["status"] == "error"
    assert "timed out" in result["reason"]
