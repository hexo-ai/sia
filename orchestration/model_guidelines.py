"""
Model-specific guidelines injected into meta-agent and feedback agent prompts.

Guidelines live in prompts/model_specific/<name>.md — add a new file to onboard a new model.
"""

import os

_MODEL_SPECIFIC_DIR = os.path.join(os.path.dirname(__file__), "prompts", "model_specific")


def _read_md(filename: str) -> str:
    path = os.path.join(_MODEL_SPECIFIC_DIR, filename)
    return open(path, encoding="utf-8").read()


def get_guidelines(model: str) -> str:
    """Return guidelines for the given model string, or empty string if none."""
    m = model.lower()

    if "gpt-oss" in m or "tinker" in m:
        return _read_md("gpt_oss.md")

    return ""
