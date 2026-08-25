"""Filesystem layout: path/filename constants and run/task path builders.

Single source of truth for the path and filename literals that were previously
scattered across orchestrator.py and context_manager.py. Path-building methods
return ``str`` (not ``Path``) to match the existing ``os.path``-based call sites,
keeping behavior byte-identical.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.resources import files as resource_files
from pathlib import Path

# Tasks that ship inside the wheel via package-data (sia/tasks/<name>/...).
BUNDLED_TASKS = ("gpqa", "lawbench", "longcot-chess", "spaceship-titanic")


class Names:
    """Every filename / relative-path literal used by a run or a task."""

    # Run / generation artifacts
    TARGET_AGENT = "target_agent.py"
    TRAIN_SCRIPT = "train.py"
    AGENT_EXECUTION_JSON = "agent_execution.json"
    AGENT_EXECUTION_DIR = "agent_execution"
    EXECUTION_GLOB = "execution_q*.json"
    STDOUT_LOG = "target_agent_stdout.log"
    TRAIN_STDOUT_LOG = "train_stdout.log"
    EVAL_LOG = "evaluation.log"
    HARBOR_RUN_LOG = "harbor_run.log"
    RESULTS_JSON = "results.json"
    CONTEXT_MD = "context.md"
    IMPROVEMENT_MD = "improvement.md"
    META_PROMPT = "meta_agent_prompt.txt"
    FEEDBACK_PROMPT = "feedback_agent_prompt.txt"
    REQUIREMENTS_TXT = "requirements.txt"
    VENV_DIR = "venv"
    RUNS_ROOT = "./runs"

    # Task inputs
    DATA_PUBLIC = "data/public"
    TASK_MD = "data/public/task.md"
    EVALUATE_PY = "evaluate.py"
    SHARED_SAMPLE_EXECUTION = "sample_agent_execution.json"
    REFERENCE_DIR = "reference"
    REFERENCE_AGENT_FILE = "reference_target_agent.py"
    SAMPLE_TASK_DESCRIPTIONS = f"{REFERENCE_DIR}/SAMPLE_TASK_DESCRIPTIONS.md"
    REFERENCE_AGENT = f"{REFERENCE_DIR}/{REFERENCE_AGENT_FILE}"
    REFERENCE_HARBOR_AGENT = "reference_harbor_agent.py"
    SHARED_DIR = "_shared"


def bundled_shared_dir() -> str:
    """Absolute path to the bundled ``sia/tasks/_shared`` directory."""
    return str(resource_files("sia.tasks") / Names.SHARED_DIR)


def venv_python_path(venv_dir: str) -> str:
    """Path to the python executable inside a venv."""
    return os.path.join(venv_dir, "bin", "python")


def venv_pip_path(venv_dir: str) -> str:
    """Path to the pip executable inside a venv."""
    return os.path.join(venv_dir, "bin", "pip")


def find_evaluate_script(task_dir: str) -> str | None:
    """Locate evaluate.py: prefer data/public/evaluate.py, then task_dir/evaluate.py, else None."""
    candidate = os.path.join(task_dir, Names.DATA_PUBLIC, Names.EVALUATE_PY)
    if os.path.exists(candidate):
        return candidate
    candidate = os.path.join(task_dir, Names.EVALUATE_PY)
    if os.path.exists(candidate):
        return candidate
    return None


def resolve_task_dir(task: str | None, task_dir: str | None) -> tuple[str, str]:
    """Resolve --task / --task_dir to a (task_dir, shared_dir) pair of real paths.

    - --task <name>  → bundled sia/tasks/<name>/, shared_dir = bundled sia/tasks/_shared/
    - --task_dir P   → P, shared_dir = P/../_shared/ if present else bundled _shared/
    """
    bundled_root = Path(str(resource_files("sia.tasks")))
    bundled_shared = bundled_root / Names.SHARED_DIR

    if task:
        resolved = bundled_root / task
        if not resolved.is_dir():
            available = ", ".join(BUNDLED_TASKS)
            raise SystemExit(f"Bundled task '{task}' not found. Available: {available}")
        return str(resolved), str(bundled_shared)

    if task_dir:
        resolved = Path(task_dir).resolve()
        if not resolved.is_dir():
            raise SystemExit(f"Task directory does not exist: {task_dir}")
        external_shared = resolved.parent / Names.SHARED_DIR
        shared = external_shared if external_shared.is_dir() else bundled_shared
        return str(resolved), str(shared)

    raise SystemExit("Either --task or --task_dir must be provided")


@dataclass(frozen=True)
class RunLayout:
    """Paths under a run directory (e.g. ``./runs/run_1``)."""

    run_dir: str

    @classmethod
    def for_run_id(cls, run_id: int, runs_root: str = Names.RUNS_ROOT) -> RunLayout:
        return cls(f"{runs_root}/run_{run_id}")

    # Generation directories: gen_dir returns an absolute path, gen_dir_rel a relative one.
    def gen_dir(self, n: int) -> str:
        return os.path.abspath(f"{self.run_dir}/gen_{n}")

    def gen_dir_rel(self, n: int) -> str:
        return os.path.join(self.run_dir, f"gen_{n}")

    @property
    def venv_dir(self) -> str:
        return os.path.join(self.run_dir, Names.VENV_DIR)

    @property
    def venv_python(self) -> str:
        return venv_python_path(self.venv_dir)

    @property
    def context_md(self) -> str:
        return os.path.join(self.run_dir, Names.CONTEXT_MD)

    def target_agent(self, n: int) -> str:
        return os.path.join(self.gen_dir(n), Names.TARGET_AGENT)

    def stdout_log(self, n: int, focus: str = "harness") -> str:
        """Return stdout log path based on improvement focus mode.

        Args:
            n: Generation number
            focus: "harness" (code/prompt improvements) or "weights" (RL-based tuning)
        """
        log_name = Names.TRAIN_STDOUT_LOG if focus == "weights" else Names.STDOUT_LOG
        return os.path.join(self.gen_dir(n), log_name)

    def improvement_md(self, n: int) -> str:
        return os.path.join(self.gen_dir(n), Names.IMPROVEMENT_MD)

    def agent_execution_dir(self, n: int) -> str:
        return os.path.join(self.gen_dir(n), Names.AGENT_EXECUTION_DIR)

    def meta_prompt(self, n: int) -> str:
        return os.path.join(self.gen_dir(n), Names.META_PROMPT)


@dataclass(frozen=True)
class TaskLayout:
    """Paths for a resolved task directory and its shared directory."""

    task_dir: str
    shared_dir: str

    @property
    def dataset_dir(self) -> str:
        return os.path.join(self.task_dir, Names.DATA_PUBLIC)

    @property
    def abs_dataset_dir(self) -> str:
        return os.path.abspath(self.dataset_dir)

    @property
    def task_md(self) -> str:
        return os.path.join(self.task_dir, Names.TASK_MD)

    @property
    def sample_descriptions(self) -> str:
        return os.path.join(self.task_dir, Names.SAMPLE_TASK_DESCRIPTIONS)

    @property
    def reference_dir(self) -> str:
        return os.path.join(self.task_dir, Names.REFERENCE_DIR)

    @property
    def reference_agent(self) -> str:
        return os.path.join(self.task_dir, Names.REFERENCE_AGENT)

    @property
    def sample_execution(self) -> str:
        return os.path.join(self.shared_dir, Names.SHARED_SAMPLE_EXECUTION)

    def evaluate_script(self) -> str | None:
        return find_evaluate_script(self.task_dir)
