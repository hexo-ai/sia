"""Characterization: lock the execution_status / execution_section text built for
the feedback prompt across the success/failure x single/multi x results matrix.
"""

import json
from pathlib import Path

from golden_master import assert_golden, normalize_paths

from sia.orchestrator import TaskFiles, _build_feedback_context

TASK_FILES = TaskFiles("desc", "ref", {}, "# Task")


def _snapshot(gen_dir, stdout_log_file, status, section) -> str:
    text = "===== EXECUTION STATUS =====\n" + status + "\n===== EXECUTION SECTION =====\n" + section
    return normalize_paths(text, {str(gen_dir): "<GEN>", str(stdout_log_file): "<LOG>"}).replace("\\", "/")


def _write_transfer_evidence(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "evaluator_status": "evaluator_completed",
                "score_key": "accuracy",
                "score_delta": 0.15,
                "reusable_bullets": [
                    "Use metric-guided rollout tuning for prompt templates.",
                    "Avoid brittle assumptions about dataset-specific fields.",
                ],
                "residue_bullets": ["Task-specific retries were introduced in this run."],
                "unsupported_claims": ["No claim about benchmark portability was validated."],
                "claim_boundary": "Treat residue as task-specific context.",
            }
        ),
        encoding="utf-8",
    )


def test_success_single_with_results(tmp_path):
    gen_dir = tmp_path / "gen_1"
    gen_dir.mkdir()
    (gen_dir / "agent_execution.json").write_text(json.dumps([{"role": "user", "content": "solve it"}]))
    (gen_dir / "results.json").write_text(json.dumps({"accuracy": 0.9, "correct": 9, "total": 10}))
    stdout_log = str(gen_dir / "target_agent_stdout.log")
    transfer_evidence_path = gen_dir / "transfer_evidence.json"
    _write_transfer_evidence(transfer_evidence_path)

    status, section = _build_feedback_context(
        current_gen=1,
        gen_dir=str(gen_dir),
        dataset_dir="/data/public",
        target_agent_success=True,
        target_agent_error_msg="",
        target_agent_stdout="line1\nline2\nline3\n",
        target_agent_stderr="",
        stdout_log_file=stdout_log,
        task_files=TASK_FILES,
        transfer_evidence_path=str(transfer_evidence_path),
    )
    assert_golden("feedback_context_success_single.txt", _snapshot(gen_dir, stdout_log, status, section))


def test_failure_single_no_results(tmp_path):
    gen_dir = tmp_path / "gen_1"
    gen_dir.mkdir()
    (gen_dir / "agent_execution.json").write_text(json.dumps([{"role": "user", "content": "attempt"}]))
    stdout_log = str(gen_dir / "target_agent_stdout.log")

    status, section = _build_feedback_context(
        current_gen=1,
        gen_dir=str(gen_dir),
        dataset_dir="/data/public",
        target_agent_success=False,
        target_agent_error_msg="Target agent failed with exit code 1",
        target_agent_stdout="boot\nrunning\ncrash\n",
        target_agent_stderr="Traceback: boom",
        stdout_log_file=stdout_log,
        task_files=TASK_FILES,
    )
    assert_golden("feedback_context_failure_single.txt", _snapshot(gen_dir, stdout_log, status, section))


def test_success_multi_with_results(tmp_path):
    gen_dir = tmp_path / "gen_1"
    exec_dir = gen_dir / "agent_execution"
    exec_dir.mkdir(parents=True)
    for i in range(2):
        (exec_dir / f"execution_q{i}.json").write_text(json.dumps([{"role": "user", "content": f"q{i}"}]))
    (gen_dir / "results.json").write_text(json.dumps({"accuracy": 0.8}))
    stdout_log = str(gen_dir / "target_agent_stdout.log")

    status, section = _build_feedback_context(
        current_gen=1,
        gen_dir=str(gen_dir),
        dataset_dir="/data/public",
        target_agent_success=True,
        target_agent_error_msg="",
        target_agent_stdout="processing q0\nprocessing q1\ndone\n",
        target_agent_stderr="",
        stdout_log_file=stdout_log,
        task_files=TASK_FILES,
    )
    assert_golden("feedback_context_success_multi.txt", _snapshot(gen_dir, stdout_log, status, section))
