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

    evaluator_status: str
    score_delta: float | None
    score_key: str | None
    reusable_bullets: list[str] = field(default_factory=list)
    residue_bullets: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    claim_boundary: str = "No evidence supports task-agnostic transfer beyond the reusable bullets above."

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluator_status": self.evaluator_status,
            "score_delta": self.score_delta,
            "score_key": self.score_key,
            "reusable_bullets": self.reusable_bullets,
            "residue_bullets": self.residue_bullets,
            "unsupported_claims": self.unsupported_claims,
            "claim_boundary": self.claim_boundary,
        }
