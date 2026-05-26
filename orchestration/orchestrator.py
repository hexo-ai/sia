"""
Directory structure (conceptual)

orchestration/
  orchestrator.py

tasks/
  task_1/
    reference/
      reference_target_agent.py
      SAMPLE_TASK_DESCRIPTIONS.md
    data/
      public/
        train.csv
        test.csv
        task.md
      private/
  task_2/
    reference/
      reference_target_agent.py
      SAMPLE_TASK_DESCRIPTIONS.md
    data/
      public/
        task.md
      private/

tasks/_shared/                 # cross-task examples/templates (public)
  sample_agent_execution.json

runs/
  run_1/ (unique meta_agent, unique feedback_agent, unique_task, reference_target_agent, config)
    gen_1: (meta_agent, reference_target_agent) -> target_agent_1 -> gen_1
    gen_2: (feedback_agent, target_agent_1) -> target_agent_2 -> gen_2
    gen_3: (feedback_agent, target_agent_2) -> target_agent_3 -> gen_3
  run_2/ (unique meta_agent, unique feedback_agent, unique_task, reference_target_agent, config)
    gen_1: (meta_agent, reference_target_agent) -> target_agent_1 -> gen_1
    gen_2: (feedback_agent, target_agent_1) -> target_agent_2 -> gen_2
    gen_3: (feedback_agent, target_agent_2) -> target_agent_3 -> gen_3
  run_3/ (unique meta_agent, unique feedback_agent, unique_task, reference_target_agent, config)
    gen_1: (meta_agent, reference_target_agent) -> target_agent_1 -> gen_1
    gen_2: (feedback_agent, target_agent_1) -> target_agent_2 -> gen_2
    gen_3: (feedback_agent, target_agent_2) -> target_agent_3 -> gen_3
"""

import os
import re
import sys
import json
import shutil
import signal
import asyncio
import logging
import argparse
import glob
import time
from pathlib import Path
from datetime import datetime

from util import run_agent
from context_manager import ContextManager
from model_guidelines import get_guidelines

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(filename: str) -> str:
    return open(os.path.join(_PROMPTS_DIR, filename), encoding="utf-8").read()


def _fill_template(template: str, **kwargs) -> str:
    """Replace {UPPER_CASE_KEY} placeholders in template. Single-pass via re.sub — safe against cascading."""
    import re
    def _replace(m):
        return str(kwargs.get(m.group(1), m.group(0)))
    return re.sub(r"\{([A-Z_][A-Z0-9_]*)\}", _replace, template)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ========================
# HELPER FUNCTIONS
# ========================

# Tracks the currently running subprocess so SIGINT/SIGTERM can kill it cleanly.
_current_proc: "subprocess.Popen | None" = None


def _kill_current_proc() -> None:
    global _current_proc
    if _current_proc is None:
        return
    try:
        os.killpg(os.getpgid(_current_proc.pid), signal.SIGKILL)
        _current_proc.wait(timeout=3)
    except (ProcessLookupError, OSError, subprocess.TimeoutExpired, ChildProcessError):
        pass
    _current_proc = None


def _signal_handler(signum, frame):
    logger.warning(f"Received signal {signum} — killing subprocess and exiting.")
    _kill_current_proc()
    sys.exit(1)


signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _run_with_timeout(command: str, timeout: int) -> tuple[int, bool]:
    """Run a shell command, killing the entire process group on timeout.

    Returns (return_code, timed_out).  Using start_new_session=True ensures
    bash and all its children (python agent, tee, etc.) share one process group
    so os.killpg reaches every orphan.
    """
    global _current_proc
    proc = subprocess.Popen(
        command,
        shell=True,
        executable="/bin/bash",
        text=True,
        start_new_session=True,
    )
    _current_proc = proc
    try:
        proc.wait(timeout=timeout)
        _current_proc = None
        return proc.returncode, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=2)
            except (ProcessLookupError, ChildProcessError):
                pass
        _current_proc = None
        return proc.returncode or -1, True


def load_agent_execution(gen_directory):
    """
    Load execution logs with automatic format detection.

    Supports two formats:
    1. Single-file: gen_X/agent_execution.json (backwards compatible)
    2. Multi-trajectory: gen_X/agent_execution/execution_q0.json, execution_q1.json, ...

    Args:
        gen_directory: Path to the generation directory

    Returns:
        tuple: (execution_data, is_multi_trajectory)
            - execution_data: dict or list containing execution log(s)
            - is_multi_trajectory: bool indicating if multi-trajectory format
    """
    execution_folder = os.path.join(gen_directory, "agent_execution")
    execution_file = os.path.join(gen_directory, "agent_execution.json")

    # Check for multi-trajectory folder first (new format)
    if os.path.isdir(execution_folder):
        logger.info(f"  → Detected multi-trajectory format (folder)")

        files = sorted(glob.glob(os.path.join(execution_folder, "execution_q*.json")))

        if not files:
            logger.warning(f"  ✗ agent_execution/ folder exists but is empty")
            return {"error": "Empty execution folder", "type": "multi-trajectory"}, True

        # Load all trajectory files
        trajectories = []
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    trajectories.append(json.load(fp))
            except json.JSONDecodeError as e:
                logger.warning(f"  ✗ Failed to parse {os.path.basename(f)}: {e}")
                trajectories.append({"error": str(e), "file": os.path.basename(f)})
            except Exception as e:
                logger.warning(f"  ✗ Error reading {os.path.basename(f)}: {e}")
                trajectories.append({"error": str(e), "file": os.path.basename(f)})

        logger.info(f"  ✓ Loaded {len(trajectories)} trajectory files")

        return {
            "trajectories": trajectories,
            "count": len(trajectories),
            "type": "multi-trajectory"
        }, True

    # Fall back to single file (old format, backwards compatible)
    elif os.path.exists(execution_file):
        logger.info(f"  → Detected single-file format")

        try:
            with open(execution_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"  ✓ Successfully loaded agent execution log")
            return data, False

        except json.JSONDecodeError as e:
            logger.warning(f"  ✗ Failed to parse agent_execution.json: {e}")
            logger.warning(f"  → The target agent may have crashed or failed to complete")

            # Return partial data for debugging
            try:
                with open(execution_file, 'r', encoding='utf-8') as f:
                    raw = f.read()
                return {
                    "error": "Parse error",
                    "raw_preview": raw[:1000],
                    "parse_error": str(e),
                    "file_size": len(raw)
                }, False
            except Exception as read_error:
                return {
                    "error": "Could not read file",
                    "read_error": str(read_error)
                }, False

        except FileNotFoundError:
            logger.error(f"  ✗ agent_execution.json not found")
            return {"error": "Execution log file not found"}, False

    # Neither exists
    else:
        logger.error(f"  ✗ No execution log found (neither file nor folder)")
        return {"error": "Execution log not found"}, False


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run the orchestrator for agent evolution')
parser.add_argument('--max_gen', type=int, default=3, help='Maximum number of generations to run (default: 3)')
parser.add_argument('--run_id', type=int, default=1, help='Run ID for this experiment (default: 1)')
parser.add_argument('--task_dir', type=str, required=True, help='Path to the task directory (e.g., ./tasks/task_1)')
parser.add_argument('--meta_model', type=str, default=None, help='Model to use for meta-agent (default: haiku for claude backend, gemini/gemini-3.1-pro-preview for openhands backend)')
parser.add_argument('--task_model', type=str, default='claude-haiku-4-5-20251001', help='Model to use for target agent (default: claude-haiku-4-5-20251001)')
parser.add_argument('--backend', type=str, default='claude', choices=['claude', 'openhands'], help='Agent backend to use: claude (Claude Code SDK) or openhands (OpenHands SDK) (default: claude)')
parser.add_argument('--target_agent_timeout', type=int, default=360, help='Hard time limit per generation in seconds — process is killed when exceeded (default: 360)')
parser.add_argument('--task_model_temperature', type=float, default=0.3, help='Sampling temperature for the task model (default: 0.3)')
parser.add_argument('--max_turns', type=int, default=30, help='Max LLM turns per target agent run (default: 30)')
parser.add_argument('--private_scores_task_models', type=str, default='',
                    help='Comma-separated models for private scoring (e.g. "openai/gpt-oss-120b,claude/..."). '
                         'Each model re-runs the target agent on the public dataset and evaluates on the private test set. '
                         'Empty or absent = skip private evaluation.')
parser.add_argument('--include_gen0', action='store_true', help='Run generation 0 using the reference_target_agent.py as-is, skipping the meta-agent entirely')
args = parser.parse_args()

# Deduplicated list of models for private scoring; empty list = skip
private_score_models: list[str] = list(dict.fromkeys(
    m.strip() for m in (args.private_scores_task_models or "").split(",") if m.strip()
))

max_gen = args.max_gen
task_dir = args.task_dir
run_id = args.run_id
backend = args.backend

# Set default meta_model based on backend if not explicitly provided
if args.meta_model is None:
    if backend == 'openhands':
        meta_model = 'gemini/gemini-3.1-pro-preview'
        logger.info("Using default OpenHands model: gemini/gemini-3.1-pro-preview")
    else:
        meta_model = 'haiku'
        logger.info("Using default Claude model: haiku")
else:
    meta_model = args.meta_model

task_model = args.task_model


def _required_api_key(model: str) -> tuple[str, list[str]]:
    """Return (description, [candidate env var names]) for a given model string."""
    m = model.lower()
    if any(m.startswith(p) for p in ("gemini/", "google/")):
        return "Gemini / Google", ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
    if "gpt-oss" in m or "tinker" in m:
        return "Tinker", ["TINKER_API_KEY"]
    if any(m.startswith(p) for p in ("claude", "anthropic/")):
        return "Anthropic", ["ANTHROPIC_API_KEY"]
    return "OpenAI", ["OPENAI_API_KEY"]


def _check_api_key(model: str) -> None:
    provider, candidates = _required_api_key(model)
    missing = [k for k in candidates if not os.environ.get(k)]
    if len(missing) == len(candidates):
        key_list = " or ".join(candidates)
        logger.error(f"Model '{model}' requires {provider} credentials — set {key_list}")
        sys.exit(1)


_check_api_key(meta_model)
_check_api_key(task_model)

logger.info(f"Configuration:")
logger.info(f"  - Maximum generations: {max_gen}")
logger.info(f"  - Task directory: {task_dir}")
logger.info(f"  - Run ID: {run_id}")
logger.info(f"  - Agent backend: {backend}")
logger.info(f"  - Meta-agent model: {meta_model}")
logger.info(f"  - Task-agent model: {task_model}")


# ========================
# SECTION 1: Load Files from Task Directory
# ========================

logger.info("Loading files from task directory...")

try:
    SAMPLE_TASK_DESCRIPTIONS = open(os.path.join(task_dir, "reference/SAMPLE_TASK_DESCRIPTIONS.md")).read()
except FileNotFoundError:
    SAMPLE_TASK_DESCRIPTIONS = ""
    logger.warning("  ⚠ reference/SAMPLE_TASK_DESCRIPTIONS.md not found — proceeding without sample descriptions")
logger.info("  ✓ Sample task descriptions loaded")

REFERENCE_TARGET_AGENT_PY = open(os.path.join(task_dir, "reference/reference_target_agent.py")).read()
logger.info("  ✓ Reference target agent loaded")

SAMPLE_AGENT_EXECUTION = json.load(open(os.path.join(task_dir, "../_shared/sample_agent_execution.json")))
logger.info("  ✓ Sample agent execution loaded")

TASK_MD = open(os.path.join(task_dir, "data/public/task.md")).read()
logger.info("  ✓ Task specification loaded")

SCAFFOLD_DESIGN_GUIDELINES = _load_prompt("scaffold_design_guidelines.md")
TARGET_AGENT_SPEC         = _load_prompt("target_agent_spec.md")
logger.info("  ✓ Shared prompt guidelines loaded")

_META_AGENT_TEMPLATE     = _load_prompt("meta_agent_prompt.md")
_FEEDBACK_AGENT_TEMPLATE = _load_prompt("feedback_agent_prompt.md")
logger.info("  ✓ Prompt templates loaded")


# ========================
# SECTION 2: Setup Run Directories
# ========================

gen_num = 0 if args.include_gen0 else 1
RUN_DIRECTORY = f"./runs/run_{run_id}"
META_AGENT_WORKING_DIRECTORY = os.path.abspath(f"{RUN_DIRECTORY}/gen_{gen_num}")
FEEDBACK_AGENT_WORKING_DIRECTORY = META_AGENT_WORKING_DIRECTORY

# Create run directory and meta_agent working directory
if os.path.exists(RUN_DIRECTORY):
    logger.error(f"Run directory already exists: {RUN_DIRECTORY}")
    logger.error("Please use a different run_id or remove the existing directory")
    sys.exit(1)

logger.info(f"Creating run directory: {RUN_DIRECTORY}")
os.makedirs(RUN_DIRECTORY, exist_ok=False)

logger.info(f"Creating meta_agent working directory: {META_AGENT_WORKING_DIRECTORY}")
os.makedirs(META_AGENT_WORKING_DIRECTORY, exist_ok=False)

# Create virtual environment
import venv
import subprocess

venv_dir = os.path.join(RUN_DIRECTORY, "venv")
logger.info(f"Creating virtual environment at: {venv_dir}")
uv_available = subprocess.run(["which", "uv"], capture_output=True).returncode == 0
if uv_available:
    subprocess.run(["uv", "venv", "--python", "3.12", venv_dir], check=True)
else:
    venv.create(venv_dir, with_pip=True)

# Path to the pip executable inside the venv
pip_executable = os.path.join(venv_dir, "bin", "pip")
def pip_install(args):
    if uv_available:
        subprocess.run(["uv", "pip", "install", "--python", pip_executable.replace("/bin/pip", "/bin/python")] + args, check=True)
    else:
        subprocess.run([pip_executable, "install"] + args, check=True)

# Install base packages from _shared/base_requirements.txt
base_requirements = os.path.abspath(os.path.join(task_dir, "../_shared/base_requirements.txt"))
logger.info(f"Installing base requirements from: {base_requirements} ({'uv pip' if uv_available else 'pip'})")
pip_install(["-r", base_requirements])

# Install task-specific requirements if present (e.g. tasks/denoising/requirements.txt)
task_requirements = os.path.join(task_dir, "requirements.txt")
if os.path.exists(task_requirements):
    logger.info(f"Installing task-specific requirements from: {task_requirements}")
    pip_install(["-r", task_requirements])
    logger.info("  ✓ Task requirements installed")

# Initialize Context Manager
logger.info("Initializing context manager...")
context_mgr = ContextManager(RUN_DIRECTORY, {
    'task_dir': task_dir,
    'meta_model': meta_model,
    'task_model': task_model,
    'backend': backend,
    'max_gen': max_gen,
})
context_mgr.initialize()
logger.info("  ✓ Context manager initialized")


# ========================
# SECTION 3: Define Prompts
# ========================

TASK_MODEL_GUIDELINES = get_guidelines(task_model)
_guidelines_section = f"\n---\n{TASK_MODEL_GUIDELINES}\n---\n" if TASK_MODEL_GUIDELINES else ""

META_AGENT_PROMPT = _fill_template(
    _META_AGENT_TEMPLATE,
    TASK_MD=TASK_MD,
    SAMPLE_TASK_DESCRIPTIONS=SAMPLE_TASK_DESCRIPTIONS,
    REFERENCE_TARGET_AGENT_PY=REFERENCE_TARGET_AGENT_PY,
    SAMPLE_AGENT_EXECUTION_JSON=json.dumps(SAMPLE_AGENT_EXECUTION, indent=2),
    META_AGENT_WORKING_DIRECTORY=META_AGENT_WORKING_DIRECTORY,
    TASK_MODEL=task_model,
    REQUIRED_API_KEYS=str(_required_api_key(task_model)[1]),
    MAX_TURNS=str(args.max_turns),
    TASK_MODEL_GUIDELINES_SECTION=_guidelines_section,
    SCAFFOLD_DESIGN_GUIDELINES=SCAFFOLD_DESIGN_GUIDELINES,
    TARGET_AGENT_SPEC=TARGET_AGENT_SPEC,
)


# ========================
# SECTION 4: Run Target Agent Creation (Meta-Agent)
# ========================

if args.include_gen0:
    import shutil
    reference_agent_src = os.path.join(task_dir, "reference/reference_target_agent.py")
    reference_agent_dst = os.path.join(META_AGENT_WORKING_DIRECTORY, "target_agent.py")
    shutil.copy(reference_agent_src, reference_agent_dst)
    logger.info(f"  ✓ Gen 0: copied reference_target_agent.py → gen_0/target_agent.py (no meta-agent)")
else:
    # Save the meta-agent prompt for debugging/transparency
    meta_agent_prompt_path = os.path.join(META_AGENT_WORKING_DIRECTORY, "meta_agent_prompt.txt")
    with open(meta_agent_prompt_path, 'w', encoding='utf-8') as f:
        f.write(META_AGENT_PROMPT)
    logger.info(f"  ✓ Saved meta-agent prompt to: {meta_agent_prompt_path}")

    MAX_META_RETRIES = 3
    for _meta_attempt in range(1, MAX_META_RETRIES + 1):
        if _meta_attempt > 1:
            logger.warning(f"  ↻ Meta-agent retry {_meta_attempt}/{MAX_META_RETRIES} — target_agent.py was not created")
        asyncio.run(run_agent(
            model_name=meta_model,
            max_turns="20",
            prompt=META_AGENT_PROMPT,
            agent_working_directory=META_AGENT_WORKING_DIRECTORY,
            backend=backend
        ))
        if Path(META_AGENT_WORKING_DIRECTORY, "target_agent.py").exists():
            logger.info(f"  ✓ target_agent.py created by meta-agent (attempt {_meta_attempt})")
            break
        logger.warning(f"  ✗ target_agent.py not found after meta-agent run (attempt {_meta_attempt}/{MAX_META_RETRIES})")
    else:
        logger.error(f"  ✗ Meta-agent failed to create target_agent.py after {MAX_META_RETRIES} attempts — aborting")
        sys.exit(1)


# ========================
# SECTION 5: Main Loop - Run Target Agent and Feedback Agent
# ========================

from pathlib import Path

# Define the dataset directory and working directory to pass as arguments
DATASET_DIRECTORY = os.path.join(task_dir, "data/public")
ABS_DATASET_DIRECTORY = os.path.abspath(DATASET_DIRECTORY)
ABS_SHARED_DIRECTORY  = os.path.abspath(os.path.join(task_dir, "../_shared"))
logger.info(f"Dataset directory: {ABS_DATASET_DIRECTORY}")
logger.info(f"Shared directory:  {ABS_SHARED_DIRECTORY}")

# Run the loop for max_gen generations
# With --include_gen0: loop starts at 0 (reference agent), feedback creates gen_1+
# Without: meta-agent already created gen_1, loop starts at 1
loop_start = 0 if args.include_gen0 else 1
for current_gen in range(loop_start, max_gen + 1):
    logger.info(f"=" * 80)
    logger.info(f"Starting Generation {current_gen} of {max_gen}")
    logger.info(f"=" * 80)

    # ========================
    # SECTION 5a: Run Target Agent
    # ========================

    current_gen_directory = os.path.abspath(f"{RUN_DIRECTORY}/gen_{current_gen}")
    target_agent_path = os.path.join(current_gen_directory, "target_agent.py")

    logger.info(f"Running target agent: {target_agent_path}")

    # Track execution results for feedback agent
    target_agent_success = True
    target_agent_stdout = ""
    target_agent_stderr = ""
    target_agent_error_msg = ""

    # Create log file paths
    stdout_log_file = os.path.join(current_gen_directory, "target_agent_stdout.log")
    stderr_log_file = os.path.join(current_gen_directory, "target_agent_stderr.log")

    logger.info(f"  → Stdout log: {stdout_log_file}")
    logger.info(f"  → Stderr log: {stderr_log_file}")
    logger.info(f"=" * 60)

    # Start timing for this generation
    generation_start_time = time.time()

    # Run target agent with real-time output using shell redirection
    try:
        python_exec = os.path.join(venv_dir, "bin", "python")
        command = (
            f"set -o pipefail; {python_exec} -u {target_agent_path} "
            f"--dataset_dir {ABS_DATASET_DIRECTORY} "
            f"--working_dir {current_gen_directory} "
            f"--shared_dir {ABS_SHARED_DIRECTORY} "
            f"--model {task_model} "
            f"--max_turns {args.max_turns} "
            f"--target_agent_timeout {args.target_agent_timeout} "
            f"--task_model_temperature {args.task_model_temperature} "
            f"2>&1 | tee {stdout_log_file}"
        )

        return_code, timed_out = _run_with_timeout(command, args.target_agent_timeout)

        try:
            with open(stdout_log_file) as f:
                target_agent_stdout = f.read()
        except Exception:
            pass
        target_agent_stderr = ""

        if timed_out:
            target_agent_success = False
            target_agent_error_msg = f"TIMEOUT — killed after {args.target_agent_timeout}s"
            logger.warning(f"  ⏱ Target agent timed out after {args.target_agent_timeout}s")
        elif return_code != 0:
            target_agent_success = False
            target_agent_error_msg = f"FAILED (exit code {return_code})"
            logger.error(f"  ✗ Target agent failed with exit code {return_code}")
        else:
            target_agent_success = True
            logger.info(f"  ✓ Generation {current_gen} completed successfully")

    except FileNotFoundError:
        logger.error(f"  ✗ Target agent not found: {target_agent_path}")
        sys.exit(1)

    except Exception as e:
        target_agent_success = False
        target_agent_error_msg = f"FAILED — {e}"
        logger.error(f"  ✗ {target_agent_error_msg}")
        try:
            with open(stdout_log_file) as f:
                target_agent_stdout = f.read()
        except Exception:
            pass

    # Calculate execution duration
    generation_duration = time.time() - generation_start_time

    # Check if improvement.md exists in current gen directory (created by previous feedback agent)
    improvement_md_path = os.path.join(current_gen_directory, "improvement.md")

    # Add generation to context (do this before feedback agent runs)
    context_mgr.add_generation(
        gen_num=current_gen,
        gen_data={
            'success': target_agent_success,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'duration': generation_duration,
            'agent_path': target_agent_path,
            'gen_dir': current_gen_directory,
            'improvement_path': improvement_md_path if os.path.exists(improvement_md_path) else None,
            'execution_type': 'Multi-trajectory' if (os.path.isdir(os.path.join(current_gen_directory, "agent_execution"))) else 'Single',
        }
    )

    # Private test scores — one run per model, display only, never fed back as signal.
    # Kept outside gen_N/ so the feedback agent cannot read them.
    private_eval = os.path.join(task_dir, "data/private/evaluate.py")
    if private_score_models and os.path.exists(private_eval):
        for priv_model in private_score_models:
            model_slug = re.sub(r"[^\w-]", "_", priv_model)[:40]
            priv_work_dir = os.path.abspath(
                os.path.join(RUN_DIRECTORY, "private_scores", f"gen_{current_gen}", model_slug)
            )
            os.makedirs(priv_work_dir, exist_ok=True)

            if priv_model == task_model:
                # Task model already ran — copy solution.py to private dir and evaluate there
                priv_solution_src = os.path.join(current_gen_directory, "solution.py")
                if not os.path.exists(priv_solution_src):
                    logger.warning(f"  [private] No solution.py in gen_{current_gen} — skipping {priv_model}")
                    continue
                shutil.copy(priv_solution_src, os.path.join(priv_work_dir, "solution.py"))
                logger.info(f"  [private] gen_{current_gen} — evaluating existing solution for task_model: {priv_model}")
            else:
                # Different model: re-run the agent in the private dir
                priv_command = (
                    f"set -o pipefail; {python_exec} -u {target_agent_path} "
                    f"--dataset_dir {ABS_DATASET_DIRECTORY} "
                    f"--working_dir {priv_work_dir} "
                    f"--shared_dir {ABS_SHARED_DIRECTORY} "
                    f"--model {priv_model} "
                    f"--max_turns {args.max_turns} "
                    f"--target_agent_timeout {args.target_agent_timeout} "
                    f"--task_model_temperature {args.task_model_temperature} "
                    f"2>&1"
                )
                logger.info(f"  [private] gen_{current_gen} — running agent with model: {priv_model}")
                _, priv_timed_out = _run_with_timeout(priv_command, args.target_agent_timeout)
                if priv_timed_out:
                    logger.warning(f"  [private] Agent timed out for {priv_model}")
                if not os.path.exists(os.path.join(priv_work_dir, "solution.py")):
                    logger.warning(f"  [private] No solution.py generated by {priv_model} — skipping private eval")
                    continue

            _run_with_timeout(
                f"{python_exec} {private_eval} --gen-dir {priv_work_dir}",
                args.target_agent_timeout,
            )
            priv_result_path = os.path.join(priv_work_dir, "private_result.json")
            if os.path.exists(priv_result_path):
                with open(priv_result_path) as f:
                    priv_result = json.load(f)
                direction = "lower" if priv_result.get("lower_is_better") else "higher"
                logger.info(f"  [private] {priv_model}: score={priv_result.get('score'):.4f} ({direction} is better)")

    # ========================
    # SECTION 5b: Run Feedback Agent (if not the last generation)
    # ========================

    if current_gen < max_gen:
        logger.info(f"Running feedback agent for generation {current_gen}")

        # Load artifacts produced by the target agent so the feedback prompt is fully populated.
        AGENT_PY = Path(current_gen_directory, "target_agent.py").read_text(encoding="utf-8")
        TASK = Path(DATASET_DIRECTORY, "task.md").read_text(encoding="utf-8")

        # Load agent execution log (supports both single-file and multi-trajectory formats)
        logger.info(f"Loading agent execution log...")
        AGENT_EXECUTION, is_multi_trajectory = load_agent_execution(current_gen_directory)

        # Build execution section for the feedback prompt
        if is_multi_trajectory:
            # Multi-trajectory format
            trajectory_count = AGENT_EXECUTION.get("count", 0)
            trajectories = AGENT_EXECUTION.get("trajectories", [])

            # Calculate success/failure counts
            # Successful trajectory = list of messages
            # Failed trajectory = dict with "error" key
            successful = sum(1 for t in trajectories if isinstance(t, list))
            failed = sum(1 for t in trajectories if isinstance(t, dict) and t.get("error"))
            # Note: failed might not equal trajectory_count - successful if there are unexpected formats

            # Show first 3 trajectories as examples
            sample_trajectories_text = ""
            for idx, traj in enumerate(trajectories[:3]):
                traj_json = json.dumps(traj, indent=2)
                # Truncate if too long
                if len(traj_json) > 1000:
                    traj_json = traj_json[:1000] + "\n  ... (truncated)"
                sample_trajectories_text += f"\n### Trajectory {idx}\n```json\n{traj_json}\n```\n"

            execution_section = f"""
**MULTI-TRAJECTORY EXECUTION**:

The agent executed {trajectory_count} separate trajectories (e.g., different questions/samples).

**Summary**:
- Total trajectories: {trajectory_count}
- Successful: {successful}
- Failed: {failed}
- Execution folder: {os.path.join(current_gen_directory, "agent_execution")}

**Sample Trajectories** (first 3 shown, you can read others from the folder):
{sample_trajectories_text}

**To analyze all trajectories**:
- Read files from: {os.path.join(current_gen_directory, "agent_execution")}
- Files named: execution_q0.json, execution_q1.json, ..., execution_q{trajectory_count-1}.json

**Analysis guidance**:
- Look for common failure patterns across trajectories
- Check if trajectories are properly isolated
- Ensure consistent behavior across all samples
"""
        else:
            # Single-trajectory format (backwards compatible)
            MAX_EXECUTION_CHARS = 80_000
            execution_json = json.dumps(AGENT_EXECUTION, indent=2)
            if len(execution_json) > MAX_EXECUTION_CHARS:
                execution_json = execution_json[:MAX_EXECUTION_CHARS] + "\n... (truncated — trajectory too large)"
            execution_section = f"""
Here is the target agent execution trajectory:
```json
{execution_json}
```

NOTE: If you see an "error" field in the above JSON, it means the execution log was malformed or missing. Focus on making the agent more robust.
"""

        # Load evaluation result (results.json produced by the target agent)
        results_json_path = os.path.join(current_gen_directory, "results.json")
        if os.path.exists(results_json_path):
            with open(results_json_path) as f:
                results_json_text = f.read()
        else:
            results_json_text = '{"error": "results.json not found — solution was not evaluated"}'

        # Prepare execution status for feedback agent
        last_lines = '\n'.join(target_agent_stdout.split('\n')[-10:])
        if target_agent_success:
            execution_status = f"SUCCESS\n\nLast output lines:\n```\n{last_lines}\n```"
        else:
            execution_status = (
                f"{target_agent_error_msg}\n\n"
                f"Last output lines:\n```\n{last_lines}\n```\n\n"
                f"Full log: {stdout_log_file}"
            )

        # Prepare next generation directory
        next_gen = current_gen + 1
        next_gen_directory = os.path.abspath(f"{RUN_DIRECTORY}/gen_{next_gen}")

        # Build previous generations list
        previous_gens_list = list(range(1, current_gen)) if current_gen > 1 else []
        previous_gens_text = ", ".join(map(str, previous_gens_list)) if previous_gens_list else "None"

        # Find the best-scoring generation so far (including current)
        best_score_so_far = -1.0
        best_gen_so_far = current_gen
        for g in range(0, current_gen + 1):
            r_path = os.path.join(RUN_DIRECTORY, f"gen_{g}", "results.json")
            if os.path.exists(r_path):
                try:
                    with open(r_path) as f:
                        r = json.load(f)
                    s = float(r.get("score", -1))
                    if s > best_score_so_far:
                        best_score_so_far = s
                        best_gen_so_far = g
                except Exception:
                    pass

        best_agent_py_path = os.path.join(RUN_DIRECTORY, f"gen_{best_gen_so_far}", "target_agent.py")
        if os.path.exists(best_agent_py_path) and best_gen_so_far != current_gen:
            best_agent_py = Path(best_agent_py_path).read_text(encoding="utf-8")
            best_agent_section = (
                f"**BEST TARGET AGENT SO FAR** (Generation {best_gen_so_far}, Score: {best_score_so_far:.4f}):\n"
                f"```python\n{best_agent_py}\n```"
            )
        else:
            best_agent_section = (
                f"**BEST TARGET AGENT SO FAR**: Generation {best_gen_so_far} "
                f"(Score: {best_score_so_far:.4f}) — same as current agent above."
            )

        # Build best-gen execution section (for prompt context)
        if best_gen_so_far != current_gen:
            best_exec_path = os.path.join(RUN_DIRECTORY, f"gen_{best_gen_so_far}", "agent_execution.json")
            if os.path.exists(best_exec_path):
                MAX_BEST_EXEC_CHARS = 80_000
                best_exec_text = Path(best_exec_path).read_text(encoding="utf-8")
                if len(best_exec_text) > MAX_BEST_EXEC_CHARS:
                    best_exec_text = best_exec_text[:MAX_BEST_EXEC_CHARS] + "\n... (truncated)"
                best_execution_section = (
                    f"**Best generation (Gen {best_gen_so_far}, Score: {best_score_so_far:.4f}) execution trajectory:**\n"
                    f"```json\n{best_exec_text}\n```"
                )
            else:
                best_execution_section = (
                    f"**Best generation (Gen {best_gen_so_far}):** agent_execution.json not found."
                )
        else:
            best_execution_section = (
                f"**Best generation (Gen {best_gen_so_far}):** same as current generation execution above."
            )

        # Build best-gen results section (for prompt context)
        if best_gen_so_far != current_gen:
            best_results_path = os.path.join(RUN_DIRECTORY, f"gen_{best_gen_so_far}", "results.json")
            if os.path.exists(best_results_path):
                best_results_text = Path(best_results_path).read_text(encoding="utf-8")
                best_results_section = (
                    f"**Best generation (Gen {best_gen_so_far}, Score: {best_score_so_far:.4f}) result:**\n"
                    f"```json\n{best_results_text}\n```"
                )
            else:
                best_results_section = (
                    f"**Best generation (Gen {best_gen_so_far}):** results.json not found."
                )
        else:
            best_results_section = (
                f"**Best generation (Gen {best_gen_so_far}):** same as current generation result above."
            )

        # Call feedback agent with full context
        feedback_agent_prompt_prepared = _fill_template(
            _FEEDBACK_AGENT_TEMPLATE,
            CURRENT_GEN=str(current_gen),
            PREVIOUS_GENS=previous_gens_text,
            CONTEXT_MD_PATH=os.path.join(RUN_DIRECTORY, "context.md"),
            CURRENT_GEN_DIR=current_gen_directory,
            TARGET_AGENT_TIMEOUT=str(args.target_agent_timeout),
            SAMPLE_TASK_DESCRIPTIONS=SAMPLE_TASK_DESCRIPTIONS,
            AGENT_PY=AGENT_PY,
            TASK=TASK,
            EXECUTION_STATUS=execution_status,
            RESULTS_JSON=results_json_text,
            EXECUTION_SECTION=execution_section,
            IMPROVEMENT_DIR=next_gen_directory,
            TASK_MODEL_GUIDELINES_SECTION=_guidelines_section,
            SCAFFOLD_DESIGN_GUIDELINES=SCAFFOLD_DESIGN_GUIDELINES,
            TARGET_AGENT_SPEC=TARGET_AGENT_SPEC,
            BEST_SCORE=f"{best_score_so_far:.4f}",
            BEST_GEN=str(best_gen_so_far),
            BEST_AGENT_SECTION=best_agent_section,
            BEST_RESULTS_SECTION=best_results_section,
            BEST_EXECUTION_SECTION=best_execution_section,
        )

        os.makedirs(next_gen_directory, exist_ok=True)

        # Save the feedback agent prompt for debugging/transparency
        feedback_prompt_path = os.path.join(next_gen_directory, "feedback_agent_prompt.txt")
        with open(feedback_prompt_path, 'w', encoding='utf-8') as f:
            f.write(feedback_agent_prompt_prepared)
        logger.info(f"  ✓ Saved feedback agent prompt to: {feedback_prompt_path}")
        MAX_FEEDBACK_RETRIES = 3
        for _feedback_attempt in range(1, MAX_FEEDBACK_RETRIES + 1):
            if _feedback_attempt > 1:
                logger.warning(f"  ↻ Feedback agent retry {_feedback_attempt}/{MAX_FEEDBACK_RETRIES} — target_agent.py was not created in gen_{next_gen}")
            asyncio.run(
                run_agent(
                    model_name=meta_model,
                    max_turns="20",
                    prompt=feedback_agent_prompt_prepared,
                    agent_working_directory=next_gen_directory,
                    backend=backend
                )
            )
            if Path(next_gen_directory, "target_agent.py").exists():
                logger.info(f"  ✓ target_agent.py created by feedback agent (attempt {_feedback_attempt})")
                break
            logger.warning(f"  ✗ target_agent.py not found after feedback agent run (attempt {_feedback_attempt}/{MAX_FEEDBACK_RETRIES})")
        else:
            logger.error(f"  ✗ Feedback agent failed to create target_agent.py for gen_{next_gen} after {MAX_FEEDBACK_RETRIES} attempts — aborting")
            sys.exit(1)

        logger.info(f"Feedback agent completed. Created improved agent for generation {next_gen}")
    else:
        logger.info(f"Generation {current_gen} is the final generation. Skipping feedback agent.")

# Finalize context with summary statistics
logger.info("Finalizing context.md with summary statistics...")
context_mgr.finalize()

from plot_utils import plot_private_scores
plot_path = plot_private_scores(RUN_DIRECTORY, task_name=os.path.basename(task_dir))
if plot_path:
    logger.info(f"Plot saved to: {plot_path}")

logger.info(f"=" * 80)
logger.info(f"Orchestrator completed all {max_gen} generations successfully!")
logger.info(f"Results saved in: {RUN_DIRECTORY}")
logger.info(f"Context summary: {os.path.join(RUN_DIRECTORY, 'context.md')}")
logger.info(f"=" * 80)
