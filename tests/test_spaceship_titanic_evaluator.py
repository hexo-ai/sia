"""Tests for the bundled Spaceship Titanic evaluator."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
EVALUATOR = REPO_ROOT / "sia" / "tasks" / "spaceship-titanic" / "data" / "public" / "evaluate.py"


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("spaceship_titanic_evaluate", EVALUATOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_default_gen_dir_output_is_results_json(monkeypatch, tmp_path):
    evaluator = _load_evaluator()
    gen_dir = tmp_path / "gen_1"
    truth_path = tmp_path / "private" / "test.csv"
    submission_path = gen_dir / "submission.csv"
    _write_csv(
        truth_path,
        "PassengerId,Transported\n"
        "0001_01,True\n"
        "0002_01,False\n",
    )
    _write_csv(
        submission_path,
        "PassengerId,Transported\n"
        "0001_01,True\n"
        "0002_01,True\n",
    )

    monkeypatch.setattr(evaluator, "default_ground_truth_path", lambda: truth_path)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--gen-dir", str(gen_dir)])

    evaluator.main()

    output_path = gen_dir / "results.json"
    assert output_path.is_file()
    results = json.loads(output_path.read_text(encoding="utf-8"))
    assert results["total_questions"] == 2
    assert results["correct"] == 1
    assert results["incorrect"] == 1
    assert results["missing"] == 0
    assert results["invalid"] == 0
    assert results["accuracy"] == pytest.approx(0.5)
    assert results["accuracy_percent"] == pytest.approx(50.0)


def test_evaluate_submission_counts_missing_invalid_and_extra(tmp_path):
    evaluator = _load_evaluator()
    truth_path = tmp_path / "private" / "test.csv"
    submission_path = tmp_path / "submission.csv"
    _write_csv(
        truth_path,
        "PassengerId,Transported\n"
        "0001_01,True\n"
        "0002_01,False\n"
        "0003_01,True\n",
    )
    _write_csv(
        submission_path,
        "PassengerId,Transported\n"
        "0001_01,True\n"
        "0002_01,maybe\n"
        "9999_99,False\n",
    )

    labels = evaluator.load_ground_truth(truth_path)
    submission = evaluator.load_submission(submission_path)
    results = evaluator.evaluate_submission(submission, labels)

    assert results["total_questions"] == 3
    assert results["correct"] == 1
    assert results["incorrect"] == 0
    assert results["missing"] == 1
    assert results["invalid"] == 1
    assert results["extra_predictions"] == 1
    assert results["accuracy"] == pytest.approx(1 / 3)
    assert results["accuracy_percent"] == pytest.approx(100 / 3)
    assert {row["status"] for row in results["details"]} == {"correct", "invalid", "missing"}
