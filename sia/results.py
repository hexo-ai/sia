"""Result dataclasses replacing positional tuple returns.

Internally the orchestrator builds these for clarity; at the call boundary it
still returns ``.as_tuple()`` to preserve the existing wire contract that tests
and callers depend on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TargetAgentResult:
    """Outcome of running a target agent generation."""

    success: bool
    stdout: str
    stderr: str
    error_msg: str

    def as_tuple(self) -> tuple[bool, str, str, str]:
        return (self.success, self.stdout, self.stderr, self.error_msg)


@dataclass
class FeedbackContext:
    """The two text blocks the feedback prompt is built from."""

    execution_status: str
    execution_section: str

    def as_tuple(self) -> tuple[str, str]:
        return (self.execution_status, self.execution_section)


@dataclass
class TransferEvidenceCard:
    """Structured output produced after each generation for feedback context and context carryover."""

    generation: int
    accepted_for_reuse: bool
    evaluator_status: str
    score_delta: float | None
    reusable_changes: list[str] = field(default_factory=list)
    task_specific_residue: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    negative_probe_hits: int = 0
    claim_boundary: str = "No evidence supports task-agnostic transfer beyond the reusable bullets above."

    def as_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "accepted_for_reuse": self.accepted_for_reuse,
            "evaluator_status": self.evaluator_status,
            "score_delta": self.score_delta,
            "reusable_changes": self.reusable_changes,
            "task_specific_residue": self.task_specific_residue,
            "unsupported_claims": self.unsupported_claims,
            "negative_probe_hits": self.negative_probe_hits,
            "claim_boundary": self.claim_boundary,
        }
