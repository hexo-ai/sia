"""Centralized configuration for SIA framework."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class Config:
    """Single source of truth for all SIA configuration defaults."""

    # Model defaults
    DEFAULT_CLAUDE_META_MODEL: str = "haiku"
    DEFAULT_OPENHANDS_META_MODEL: str = "gemini/gemini-3.1-pro-preview"
    DEFAULT_TASK_MODEL: str = "claude-haiku-4-5-20251001"

    # Generation defaults
    DEFAULT_MAX_GENERATIONS: int = 3
    DEFAULT_RUN_ID: int = 1

    # Agent execution
    DEFAULT_MAX_TURNS: int = 20
    CONTEXT_SUMMARY_MAX_TURNS: int = 5
    DEFAULT_BACKEND: str = "claude"

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

    # Virtual environment packages.
    VENV_PACKAGES: ClassVar[list[str]] = [
        "anthropic",
        "openai",
        "python-dotenv",
        "google-genai",
        "tqdm",
        "pydantic",
        "scikit-learn",
        "pandas",
        "numpy",
    ]

    @classmethod
    def from_env(cls) -> Config:
        """Create Config with overrides from SIA_* environment variables."""
        cfg = cls()
        env_map = {
            "SIA_META_MODEL": ("DEFAULT_CLAUDE_META_MODEL", str),
            "SIA_TASK_MODEL": ("DEFAULT_TASK_MODEL", str),
            "SIA_MAX_GENERATIONS": ("DEFAULT_MAX_GENERATIONS", int),
            "SIA_BACKEND": ("DEFAULT_BACKEND", str),
            "SIA_MAX_TURNS": ("DEFAULT_MAX_TURNS", int),
            "SIA_SANDBOX_MODE": ("SANDBOX_MODE", str),
        }
        for env_var, (attr, converter) in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                with contextlib.suppress(ValueError, TypeError):
                    setattr(cfg, attr, converter(val))
        return cfg
