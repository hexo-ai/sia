"""Validate that all task directories have the required structure."""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
TASKS_DIR = REPO_ROOT / "sia" / "tasks"

# Directories that aren't actual tasks
SKIP_DIRS = {"_shared"}


def _task_dirs():
    """Yield task directory paths (skip _shared and hidden dirs)."""
    for p in sorted(TASKS_DIR.iterdir()):
        if p.is_dir() and p.name not in SKIP_DIRS and not p.name.startswith("."):
            yield p


@pytest.fixture(params=list(_task_dirs()), ids=lambda p: p.name)
def task_dir(request):
    return request.param


def test_task_has_public_data(task_dir):
    public = task_dir / "data" / "public"
    assert public.is_dir(), f"{task_dir.name}: missing data/public/"


def test_task_has_task_md(task_dir):
    task_md = task_dir / "data" / "public" / "task.md"
    assert task_md.is_file(), f"{task_dir.name}: missing data/public/task.md"
    content = task_md.read_text()
    assert len(content) > 50, f"{task_dir.name}: task.md is too short"


def test_task_has_reference_dir(task_dir):
    ref = task_dir / "reference"
    assert ref.is_dir(), f"{task_dir.name}: missing reference/"


def test_task_has_reference_agent(task_dir):
    agent = task_dir / "reference" / "reference_target_agent.py"
    assert agent.is_file(), f"{task_dir.name}: missing reference/reference_target_agent.py"


def test_task_has_sample_descriptions(task_dir):
    desc = task_dir / "reference" / "SAMPLE_TASK_DESCRIPTIONS.md"
    assert desc.is_file(), f"{task_dir.name}: missing reference/SAMPLE_TASK_DESCRIPTIONS.md"


def test_shared_sample_execution_exists():
    sample = TASKS_DIR / "_shared" / "sample_agent_execution.json"
    assert sample.is_file(), "Missing tasks/_shared/sample_agent_execution.json"
    data = json.loads(sample.read_text())
    assert isinstance(data, (list, dict)), "sample_agent_execution.json must be valid JSON"
