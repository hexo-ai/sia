"""Characterization: lock the full context.md produced by ContextManager.

Drives the manager through a realistic two-generation run (with metrics, deltas,
and improvement.md insights) and snapshots the entire context.md. _generate_llm_summary
is patched to None so the output is deterministic (no network / no LLM section).
"""

import json
from unittest.mock import patch

from golden_master import assert_golden, normalize_timestamps

from sia.context_manager import ContextManager

GEN1_AGENT = "print('gen 1 agent')\n"
# Larger gen-2 file so size/line deltas are exercised.
GEN2_AGENT = "import sys\n\n\ndef main():\n    print('gen 2 agent, improved')\n\n\nmain()\n"
IMPROVEMENT_MD = (
    "# Improvement Plan\n\n"
    "- Added structured error handling so the agent recovers from tool failures gracefully.\n"
    "- Switched to a retry loop with exponential backoff for transient API errors.\n"
    "- Improved logging to capture each tool call and its result for later analysis.\n"
)
TRANSFER_EVIDENCE = {
    "generation": 2,
    "accepted_for_reuse": True,
    "evaluator_status": "passed",
    "score_delta": 25.0,
    "reusable_changes": [
        "Added structured error handling so the agent recovers from tool failures gracefully.",
        "Improved logging to capture each tool call and its result for later analysis.",
    ],
    "task_specific_residue": ["The retry loop is task-specific to this evaluation harness."],
    "unsupported_claims": ["No benchmark-portable claim was validated in this run."],
    "negative_probe_hits": 0,
    "claim_boundary": "Treat residue as task-specific context.",
}


@patch("sia.context_manager.ContextManager._generate_llm_summary", return_value=None)
def test_context_md_golden(_mock_llm, tmp_path):
    run_dir = tmp_path / "run_1"
    gen1 = run_dir / "gen_1"
    gen2 = run_dir / "gen_2"
    gen1.mkdir(parents=True)
    gen2.mkdir(parents=True)

    (gen1 / "target_agent.py").write_text(GEN1_AGENT)
    (gen2 / "target_agent.py").write_text(GEN2_AGENT)
    (gen2 / "improvement.md").write_text(IMPROVEMENT_MD)
    (gen1 / "results.json").write_text(json.dumps({"accuracy": 50.0, "correct": 99, "total": 198}))
    (gen2 / "results.json").write_text(json.dumps({"accuracy": 75.0, "correct": 148, "total": 198}))
    (gen2 / "transfer_evidence.json").write_text(json.dumps(TRANSFER_EVIDENCE))

    cm = ContextManager(
        str(run_dir),
        {
            "task_dir": "/tasks/example",
            "meta_model": "haiku",
            "task_model": "claude-haiku-4-5-20251001",
            "agent_impl": "claude",
            "max_gen": 2,
        },
    )
    cm.initialize()
    cm.add_generation(
        1,
        {
            "success": True,
            "timestamp": "2026-01-01 00:00:00",
            "duration": 1.5,
            "agent_path": str(gen1 / "target_agent.py"),
            "gen_dir": str(gen1),
            "improvement_path": None,
            "execution_type": "Single",
        },
    )
    cm.add_generation(
        2,
        {
            "success": True,
            "timestamp": "2026-01-01 00:05:00",
            "duration": 2.5,
            "agent_path": str(gen2 / "target_agent.py"),
            "gen_dir": str(gen2),
            "improvement_path": str(gen2 / "improvement.md"),
            "transfer_evidence_path": str(gen2 / "transfer_evidence.json"),
            "execution_type": "Single",
        },
    )
    cm.finalize()

    content = normalize_timestamps((run_dir / "context.md").read_text(encoding="utf-8"))
    assert_golden("context.md", content)
