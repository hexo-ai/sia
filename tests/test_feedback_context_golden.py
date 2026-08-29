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
                "generation": 1,
                "accepted_for_reuse": True,
                "evaluator_status": "passed",
                "score_delta": 0.15,
                "reusable_changes": [
                    "Use metric-guided rollout tuning for prompt templates.",
                    "Avoid brittle assumptions about dataset-specific fields.",
                ],
                "task_specific_residue": ["Task-specific retries were introduced in this run."],
                "unsupported_claims": ["No claim about benchmark portability was validated."],
                "negative_probe_hits": 0,
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


def test_malformed_transfer_evidence_does_not_break_context(tmp_path):
    gen_dir = tmp_path / "gen_1"
    gen_dir.mkdir()
    (gen_dir / "agent_execution.json").write_text(json.dumps([{"role": "user", "content": "attempt"}]))
    transfer_evidence_path = gen_dir / "transfer_evidence.json"
    transfer_evidence_path.write_text(
        json.dumps(
            {
                "generation": 1,
                "accepted_for_reuse": False,
                "evaluator_status": "missing",
                "reusable_changes": [1, "valid reusable change"],
                "task_specific_residue": [None, "Task-specific branch"],
                "unsupported_claims": [2, "Missing evidence"],
                "negative_probe_hits": 1,
                "claim_boundary": "Stay conservative.",
            }
        ),
        encoding="utf-8",
    )
    stdout_log = str(gen_dir / "target_agent_stdout.log")

    status, section = _build_feedback_context(
        current_gen=1,
        gen_dir=str(gen_dir),
        dataset_dir="/data/public",
        target_agent_success=True,
        target_agent_error_msg="",
        target_agent_stdout="line1\nline2\n",
        target_agent_stderr="",
        stdout_log_file=stdout_log,
        task_files=TASK_FILES,
        transfer_evidence_path=str(transfer_evidence_path),
    )

    snapshot = _snapshot(gen_dir, stdout_log, status, section)
    assert "**TRANSFER EVIDENCE**: evaluator=missing" in snapshot
    assert "valid reusable change" in snapshot
    assert "Task-specific branch" in snapshot
    assert "Missing evidence" in snapshot


def test_negative_delta_transfer_evidence_is_not_rendered_as_reusable(tmp_path):
    gen_dir = tmp_path / "gen_2"
    gen_dir.mkdir()
    (gen_dir / "agent_execution.json").write_text(json.dumps([{"role": "user", "content": "attempt"}]))
    transfer_evidence_path = gen_dir / "transfer_evidence.json"
    transfer_evidence_path.write_text(
        json.dumps(
            {
                "generation": 2,
                "accepted_for_reuse": False,
                "evaluator_status": "passed",
                "score_delta": -0.4,
                "reusable_changes": ["Reusable-looking change"],
                "task_specific_residue": ["Task-specific fallback"],
                "unsupported_claims": [],
                "negative_probe_hits": 0,
                "claim_boundary": "Do not carry negative-delta changes forward.",
            }
        ),
        encoding="utf-8",
    )
    stdout_log = str(gen_dir / "target_agent_stdout.log")

    status, section = _build_feedback_context(
        current_gen=2,
        gen_dir=str(gen_dir),
        dataset_dir="/data/public",
        target_agent_success=True,
        target_agent_error_msg="",
        target_agent_stdout="line1\nline2\n",
        target_agent_stderr="",
        stdout_log_file=stdout_log,
        task_files=TASK_FILES,
        transfer_evidence_path=str(transfer_evidence_path),
    )

    snapshot = _snapshot(gen_dir, stdout_log, status, section)
    assert "Accepted for reuse: no" in snapshot
    assert "Candidate changes not accepted for reuse" in snapshot
    assert "Accepted reusable changes" not in snapshot
