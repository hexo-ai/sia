"""Centralized configuration for SIA framework."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import ClassVar


def _to_str_tuple(value: str) -> tuple[str, ...]:
    """Parse a comma-separated env-var string into a tuple of trimmed strings."""
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _to_bool(value: str) -> bool:
    """Parse a truthy/falsy env-var string ("1"/"true"/"yes"/"on" -> True)."""
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    """Single source of truth for all SIA configuration defaults."""

    # Agent profile defaults (JSON profiles selected on the CLI, see sia/defaults/profiles/)
    DEFAULT_META_AGENT_PROFILE: str = "default-meta"
    DEFAULT_TARGET_AGENT_PROFILE: str = "default-target"

    # Model defaults (fallbacks for context metadata / env overrides)
    DEFAULT_CLAUDE_META_MODEL: str = "haiku"
    DEFAULT_OPENHANDS_META_MODEL: str = "gemini/gemini-3.1-pro-preview"
    DEFAULT_TASK_MODEL: str = "claude-haiku-4-5-20251001"

    # Generation defaults
    DEFAULT_MAX_GENERATIONS: int = 3
    DEFAULT_RUN_ID: int = 1

    # Agent execution
    DEFAULT_MAX_TURNS: int = 20
    CONTEXT_SUMMARY_MAX_TURNS: int = 5
    DEFAULT_AGENT_IMPL: str = "claude"

    # Truncation limits
    AGENT_CODE_PREVIEW_LIMIT: int = 3000
    TRAJECTORY_PREVIEW_LIMIT: int = 1000
    TOOL_RESULT_PREVIEW_LIMIT: int = 500
    INSIGHT_PREVIEW_LIMIT: int = 200

    # Timeouts
    SHELL_TIMEOUT: int = 30
    EVAL_TIMEOUT: int = 600

    # Sandbox settings
    SANDBOX_MODE: str = "none"  # "none" or "docker"
    DOCKER_IMAGE: str = "python:3.11-slim"
    DOCKER_MEMORY_LIMIT: str = "2g"
    DOCKER_CPU_LIMIT: float = 2.0
    DOCKER_TIMEOUT: int = 3600  # seconds

    # File size limits (bytes)
    MAX_CONTEXT_FILE_SIZE: int = 10_000_000  # 10 MB
    MAX_EXECUTION_LOG_SIZE: int = 50_000_000  # 50 MB

    # Verifier->feedback contract. Pass-set: a grader item whose `status` is not in
    # this set counts as a failure when building the gold-free eval summary.
    VERIFIER_PASS_STATUSES: tuple[str, ...] = ("CORRECT", "PASS", "correct")
    # Cap on the number of FAILED held-out items surfaced (reference-answer-free) in
    # the feedback prompt's curated eval summary. Diversified across status and group.
    FEEDBACK_FAILURE_SAMPLES: int = 20

    # Held-out-integrity seal — impl-agnostic OS-level backstop. While a meta/feedback
    # agent runs, strip all permissions on the task's data/private dir so a read fails at
    # the filesystem layer, not just via the prompt notice + claude-impl PreToolUse hook.
    # Applied only around the sequential meta/feedback agent calls — never during grading,
    # which legitimately reads the dir. Set SIA_PRIVATE_DIR_GUARD=0 to disable.
    PRIVATE_DIR_GUARD: bool = True

    # Virtual environment packages.
    VENV_PACKAGES: ClassVar[list[str]] = [
        "anthropic",
        "openai",
        "python-dotenv",
        "google-genai",
        "claude-agent-sdk",
        "tqdm",
        "pydantic",
        "scikit-learn",
        "pandas",
        "numpy",
    ]

    # Additional packages only for weights mode (RL training)
    WEIGHTS_VENV_PACKAGES: ClassVar[list[str]] = [
        "vllm",
        "tinker",
        "tinker-cookbook[modal] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly",
    ]

    @classmethod
    def from_env(cls) -> Config:
        """Create Config with overrides from SIA_* environment variables."""
        cfg = cls()
        env_map = {
            "SIA_META_AGENT_PROFILE": ("DEFAULT_META_AGENT_PROFILE", str),
            "SIA_TARGET_AGENT_PROFILE": ("DEFAULT_TARGET_AGENT_PROFILE", str),
            "SIA_META_MODEL": ("DEFAULT_CLAUDE_META_MODEL", str),
            "SIA_TASK_MODEL": ("DEFAULT_TASK_MODEL", str),
            "SIA_MAX_GENERATIONS": ("DEFAULT_MAX_GENERATIONS", int),
            "SIA_AGENT_IMPL": ("DEFAULT_AGENT_IMPL", str),
            "SIA_MAX_TURNS": ("DEFAULT_MAX_TURNS", int),
            "SIA_SANDBOX_MODE": ("SANDBOX_MODE", str),
            "SIA_VERIFIER_PASS_STATUSES": ("VERIFIER_PASS_STATUSES", _to_str_tuple),
            "SIA_FEEDBACK_FAILURE_SAMPLES": ("FEEDBACK_FAILURE_SAMPLES", int),
            "SIA_PRIVATE_DIR_GUARD": ("PRIVATE_DIR_GUARD", _to_bool),
        }
        for env_var, (attr, converter) in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                with contextlib.suppress(ValueError, TypeError):
                    setattr(cfg, attr, converter(val))
        return cfg
