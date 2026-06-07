"""Tests for the bundled GPQA evaluator contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
GPQA_EVALUATOR = REPO_ROOT / "sia" / "tasks" / "gpqa" / "data" / "public" / "evaluate.py"


def _load_gpqa_evaluator():
    spec = importlib.util.spec_from_file_location("gpqa_evaluate", GPQA_EVALUATOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_gen_dir_output_is_results_json(monkeypatch, tmp_path):
    evaluator = _load_gpqa_evaluator()
    gen_dir = tmp_path / "gen_1"
    gen_dir.mkdir()
    submission_path = gen_dir / "submission.json"
    submission_path.write_text('{"answers": {"1": "A"}}', encoding="utf-8")

    monkeypatch.setattr(evaluator, "load_ground_truth", lambda _path: [{"id": 1, "correct_answer_letter": "A"}])
    monkeypatch.setattr(evaluator, "find_submission_file", lambda _gen_dir: submission_path)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--gen-dir", str(gen_dir)])

    evaluator.main()

    assert (gen_dir / "results.json").is_file()
    assert not (gen_dir / "evaluation_results.json").exists()
