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

import argparse
import asyncio
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
import venv
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files as resource_files
from pathlib import Path

from sia import __version__
from sia.config import Config
from sia.context_manager import ContextManager
from sia.util import AgentBackend, run_agent

# Tasks that ship inside the wheel via package-data (sia/tasks/<name>/...).
BUNDLED_TASKS = ("gpqa", "lawbench", "longcot-chess", "spaceship-titanic")


def resolve_task_dir(task: str | None, task_dir: str | None) -> tuple[str, str]:
    """Resolve --task / --task_dir to a (task_dir, shared_dir) pair of real paths.

    - --task <name>  → bundled sia/tasks/<name>/, shared_dir = bundled sia/tasks/_shared/
    - --task_dir P   → P, shared_dir = P/../_shared/ if present else bundled _shared/
    """
    bundled_root = Path(str(resource_files("sia.tasks")))
    bundled_shared = bundled_root / "_shared"

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
        external_shared = resolved.parent / "_shared"
        shared = external_shared if external_shared.is_dir() else bundled_shared
        return str(resolved), str(shared)

    raise SystemExit("Either --task or --task_dir must be provided")


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


# ========================
# HELPER FUNCTIONS
# ========================


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
        logger.info("  → Detected multi-trajectory format (folder)")

        files = sorted(glob.glob(os.path.join(execution_folder, "execution_q*.json")))

        if not files:
            logger.warning("  ✗ agent_execution/ folder exists but is empty")
            return {"error": "Empty execution folder", "type": "multi-trajectory"}, True

        # Load all trajectory files
        trajectories = []
        for f in files:
            try:
                file_size = os.path.getsize(f)
                if file_size > Config.MAX_EXECUTION_LOG_SIZE:
                    logger.warning(f"Skipping oversized trajectory ({file_size:,} bytes): {os.path.basename(f)}")
                    trajectories.append({"error": "File too large", "file": os.path.basename(f), "size": file_size})
                    continue
                with open(f, encoding="utf-8") as fp:
                    trajectories.append(json.load(fp))
            except json.JSONDecodeError as e:
                logger.warning(f"  ✗ Failed to parse {os.path.basename(f)}: {e}")
                trajectories.append({"error": str(e), "file": os.path.basename(f)})
            except (OSError, KeyError) as e:
                logger.warning(f"  ✗ Error reading {os.path.basename(f)}: {e}")
                trajectories.append({"error": str(e), "file": os.path.basename(f)})

        logger.info(f"  ✓ Loaded {len(trajectories)} trajectory files")

        return {"trajectories": trajectories, "count": len(trajectories), "type": "multi-trajectory"}, True

    # Fall back to single file (old format, backwards compatible)
    elif os.path.exists(execution_file):
        logger.info("  → Detected single-file format")

        try:
            file_size = os.path.getsize(execution_file)
            if file_size > Config.MAX_EXECUTION_LOG_SIZE:
                logger.warning(f"Execution log too large ({file_size:,} bytes), skipping")
                return {"error": "File too large", "size": file_size}, False
            with open(execution_file, encoding="utf-8") as f:
                data = json.load(f)
            logger.info("  ✓ Successfully loaded agent execution log")
            return data, False

        except json.JSONDecodeError as e:
            logger.warning(f"  ✗ Failed to parse agent_execution.json: {e}")
            logger.warning("  → The target agent may have crashed or failed to complete")

            # Return partial data for debugging
            try:
                with open(execution_file, encoding="utf-8") as f:
                    raw = f.read()
                return {
                    "error": "Parse error",
                    "raw_preview": raw[:1000],
                    "parse_error": str(e),
                    "file_size": len(raw),
                }, False
            except OSError as read_error:
                return {"error": "Could not read file", "read_error": str(read_error)}, False

        except FileNotFoundError:
            logger.error("  ✗ agent_execution.json not found")
            return {"error": "Execution log file not found"}, False

    # Neither exists
    else:
        logger.error("  ✗ No execution log found (neither file nor folder)")
        return {"error": "Execution log not found"}, False


def run_evaluation(gen_directory, task_dir, venv_dir):
    """
    Run evaluate.py if it exists in the task's public data directory.

    Args:
        gen_directory: Path to the generation directory containing submission files
        task_dir: Path to the task directory
        venv_dir: Path to the virtual environment

    Returns:
        dict: Evaluation results or error information
    """
    # Look for evaluate.py in data/public/ first, then fall back to task_dir
    evaluate_script = os.path.join(task_dir, "data/public/evaluate.py")
    if not os.path.exists(evaluate_script):
        evaluate_script = os.path.join(task_dir, "evaluate.py")

    # Check if evaluate.py exists
    if not os.path.exists(evaluate_script):
        logger.info(f"  → No evaluate.py found in {task_dir}, skipping evaluation")
        return {"status": "skipped", "reason": "evaluate.py not found"}

    logger.info(f"Running evaluation script: {evaluate_script}")

    # Create evaluation log file
    eval_log_file = os.path.join(gen_directory, "evaluation.log")
    logger.info(f"  → Evaluation log: {eval_log_file}")

    # Run evaluate.py as subprocess with --gen-dir
    try:
        python_exec = os.path.join(venv_dir, "bin", "python")
        result = subprocess.run(
            [python_exec, evaluate_script, "--gen-dir", gen_directory],
            capture_output=True,
            text=True,
            timeout=Config.EVAL_TIMEOUT,
        )
        # Write combined output to log file
        eval_output = result.stdout + result.stderr
        Path(eval_log_file).write_text(eval_output, encoding="utf-8")

        if result.returncode != 0:
            logger.error(f"  ✗ Evaluation failed with exit code {result.returncode}")
            return {
                "status": "error",
                "reason": f"evaluate.py exited with code {result.returncode}",
                "log_path": eval_log_file,
                "output": eval_output,
            }

        # Check if results.json was created
        results_json_path = os.path.join(gen_directory, "results.json")
        if os.path.exists(results_json_path):
            logger.info("  ✓ Evaluation completed successfully")
            logger.info(f"  ✓ Results saved to: {results_json_path}")

            # Load and log results
            try:
                with open(results_json_path) as f:
                    results = json.load(f)
                logger.info(f"    Results: {json.dumps(results, indent=2)}")
            except (json.JSONDecodeError, OSError):
                pass

            return {
                "status": "success",
                "log_path": eval_log_file,
                "results_path": results_json_path,
                "output": eval_output,
            }
        else:
            logger.warning("  ⚠ Evaluation completed but results.json not found")
            return {
                "status": "warning",
                "reason": "results.json not created by evaluate.py",
                "log_path": eval_log_file,
                "output": eval_output,
            }

    except subprocess.TimeoutExpired:
        logger.error(f"  ✗ Evaluation timed out after {Config.EVAL_TIMEOUT}s")
        return {"status": "error", "reason": f"Evaluation timed out after {Config.EVAL_TIMEOUT}s"}
    except (subprocess.SubprocessError, OSError) as e:
        logger.error(f"  ✗ Unexpected error during evaluation: {e}")
        logger.error(traceback.format_exc())
        return {"status": "error", "reason": str(e), "traceback": traceback.format_exc()}


def _print_welcome():
    banner = rf"""
     _______. __       ___
    /       ||  |     /   \
   |   (----`|  |    /  ^  \
    \   \    |  |   /  /_\  \
.----)   |   |  |  /  _____  \
|_______/    |__| /__/     \__\

    Self-Improving AI framework

    • Version : v{__version__}
    • Docs    : https://github.com/hexo-ai/sia
    • Help    : sia --help
"""
    print(banner)


def _run_target_agent_sandboxed(
    python_exec: str,
    target_agent_path: str,
    dataset_dir: str,
    working_dir: str,
    stdout_log_file: str,
    config: Config,
) -> int:
    """Run target agent inside a Docker container for isolation.

    Mounts dataset_dir as read-only and working_dir as read-write.
    Network access is disabled.
    """
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--memory",
        config.DOCKER_MEMORY_LIMIT,
        f"--cpus={config.DOCKER_CPU_LIMIT}",
        "-v",
        f"{dataset_dir}:/data:ro",
        "-v",
        f"{working_dir}:/work:rw",
        config.DOCKER_IMAGE,
        "python",
        "-u",
        "/work/target_agent.py",
        "--dataset_dir",
        "/data",
        "--working_dir",
        "/work",
    ]

    with open(stdout_log_file, "w", encoding="utf-8") as log_fh:
        process = subprocess.Popen(
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in process.stdout:
            print(line, end="")
            log_fh.write(line)
        return process.wait()


@dataclass
class TaskFiles:
    """Container for task reference files loaded from disk."""

    sample_task_descriptions: str
    reference_target_agent_py: str
    sample_agent_execution: dict
    task_md: str


def load_task_files(task_dir: str, shared_dir: str) -> TaskFiles:
    """Load all reference files from the task directory."""
    logger.info("Loading files from task directory...")

    sample_task_descriptions = Path(task_dir, "reference/SAMPLE_TASK_DESCRIPTIONS.md").read_text()
    logger.info("  ✓ Sample task descriptions loaded")

    reference_target_agent_py = Path(task_dir, "reference/reference_target_agent.py").read_text()
    logger.info("  ✓ Reference target agent loaded")

    with open(os.path.join(shared_dir, "sample_agent_execution.json")) as f:
        sample_agent_execution = json.load(f)
    logger.info("  ✓ Sample agent execution loaded")

    task_md = Path(task_dir, "data/public/task.md").read_text()
    logger.info("  ✓ Task specification loaded")

    return TaskFiles(
        sample_task_descriptions=sample_task_descriptions,
        reference_target_agent_py=reference_target_agent_py,
        sample_agent_execution=sample_agent_execution,
        task_md=task_md,
    )


@dataclass
class RunSetup:
    """Container for run directory paths and managers."""

    run_directory: str
    meta_agent_working_directory: str
    venv_dir: str
    context_mgr: ContextManager


def _create_venv(venv_dir: str, packages: list[str]) -> None:
    """Create a virtual environment and install packages."""
    if shutil.which("uv"):
        subprocess.run(["uv", "venv", venv_dir], check=True)
        subprocess.run(
            ["uv", "pip", "install", "--python", os.path.join(venv_dir, "bin", "python"), *packages],
            check=True,
        )
    else:
        venv.create(venv_dir, with_pip=True)
        subprocess.run([os.path.join(venv_dir, "bin", "pip"), "install", *packages], check=True)


def setup_run_directory(
    run_id: int,
    task_dir: str,
    meta_model: str,
    task_model: str,
    backend: AgentBackend,
    max_gen: int,
) -> RunSetup:
    """Create run directories, venv, and context manager."""
    gen_num = 1
    run_directory = f"./runs/run_{run_id}"
    meta_agent_working_directory = os.path.abspath(f"{run_directory}/gen_{gen_num}")

    if os.path.exists(run_directory):
        logger.error(f"Run directory already exists: {run_directory}")
        logger.error("Please use a different run_id or remove the existing directory")
        sys.exit(1)

    logger.info(f"Creating run directory: {run_directory}")
    os.makedirs(run_directory, exist_ok=False)

    logger.info(f"Creating meta_agent working directory: {meta_agent_working_directory}")
    os.makedirs(meta_agent_working_directory, exist_ok=False)

    venv_dir = os.path.join(run_directory, "venv")
    logger.info(f"Creating virtual environment at: {venv_dir}")
    _create_venv(venv_dir, Config.VENV_PACKAGES)

    logger.info("Initializing context manager...")
    context_mgr = ContextManager(
        run_directory,
        {
            "task_dir": task_dir,
            "meta_model": meta_model,
            "task_model": task_model,
            "backend": backend,
            "max_gen": max_gen,
        },
    )
    context_mgr.initialize()
    logger.info("  ✓ Context manager initialized")

    return RunSetup(
        run_directory=run_directory,
        meta_agent_working_directory=meta_agent_working_directory,
        venv_dir=venv_dir,
        context_mgr=context_mgr,
    )


def build_meta_prompt(
    task_files: TaskFiles,
    task_model: str,
    working_dir: str,
) -> str:
    """Build the meta-agent prompt for creating the initial target agent."""
    return f"""You are a meta-agent. Your task is to create a target agent which can execute a task. Go ahead and create a target_agent.py for the target agent, which in turn can solve the given task.

Here is the FULL TASK SPECIFICATION that your target_agent.py will need to solve:
{task_files.task_md}

Here are a couple of sample task descriptions which the target agent has to solve:
{task_files.sample_task_descriptions}

Here is a sample target_agent.py showing the complete implementation pattern (READ THE ENTIRE FILE):
{task_files.reference_target_agent_py}

Here is a sample agent execution trajectory:
{json.dumps(task_files.sample_agent_execution, indent=2)}

CRITICAL RULES - FOLLOW EXACTLY:

1. The current working directory is {working_dir}. Create the target_agent.py in the current working directory itself.

2. The target_agent.py MUST accept two command-line arguments:
   - --dataset_dir: Absolute path to the dataset directory (READ-ONLY, provided at runtime)
   - --working_dir: Absolute path to the working directory (READ-WRITE, provided at runtime)

3. CRITICAL: The target_agent.py must INCLUDE these paths in the prompt it sends to {task_model}. {task_model} MUST be explicitly told:
   - Where the dataset directory is located (the exact path from --dataset_dir)
   - Where the working directory is located (the exact path from --working_dir)
   - That it can ONLY READ from the dataset directory
   - That it can READ from and WRITE to the working directory

   DO NOT let {task_model} search for data in random locations. The prompt must say: "The dataset is at: <actual_dataset_dir_path>"

4. The target agent can ONLY read from the dataset directory provided via --dataset_dir, and can ONLY write to the working directory specified by --working_dir. It must NOT access any other directories on the filesystem.

5. EXECUTION LOGGING - CRITICAL:

   The target_agent.py must log its execution trajectory properly. The format depends on the task type:

   **FOR TASKS WITH MULTIPLE INDEPENDENT SAMPLES** (e.g., GPQA with 198 questions, multiple test cases):
   - Create a folder: agent_execution/ in the working directory
   - Save each sample separately: execution_q0.json, execution_q1.json, execution_q2.json, etc.
   - Each file contains the complete trajectory for that ONE sample only
   - Files must be named sequentially: execution_q0.json, execution_q1.json, ...

   **FOR TASKS WITH SINGLE EXECUTION** (e.g., building one ML model, analyzing one dataset):
   - Save to a single file: agent_execution.json in the working directory
   - File contains the complete execution trajectory

   **HOW TO DETERMINE WHICH FORMAT**:
   - Read the task description carefully
   - If it mentions "independent items", "dataset with multiple records to process separately"
     → Use multi-trajectory (folder with multiple files)
   - If it's about "build a model", "analyze the dataset", "create one solution", "optimize one system"
     → Use single-trajectory (one JSON file)

   **FORMAT REQUIREMENTS** (both formats):
   - Use the same format as the sample agent execution trajectory provided above
   - Include all messages, tool calls, and their results
   - Ensure valid JSON (properly close all arrays/objects)
   - Make sure to properly close the JSON file(s) to avoid corruption

6. Do NOT attempt to write to or modify files inside the dataset directory. It is READ-ONLY.
7. The target_agent.py should use only the "{task_model}" model when invoking the language model (do not use any other model).
8. DO NOT hardcode any specific dataset paths in the target_agent.py code. The paths will be provided at runtime via command-line arguments and MUST be passed to {task_model} in the prompt.

Example invocation (paths will vary at runtime):
    python target_agent.py --dataset_dir /path/to/dataset --working_dir /path/to/working
"""


def _run_target_agent(
    venv_dir: str,
    target_agent_path: str,
    abs_dataset_dir: str,
    gen_dir: str,
    stdout_log_file: str,
    sandbox: str,
    env_config: Config,
) -> tuple[bool, str, str, str]:
    """Run the target agent subprocess.

    Returns (success, stdout, stderr, error_msg).
    """
    python_exec = os.path.join(venv_dir, "bin", "python")
    return_code = 0
    target_agent_stdout = ""
    target_agent_stderr = ""
    target_agent_error_msg = ""

    try:
        if sandbox == "docker":
            return_code = _run_target_agent_sandboxed(
                python_exec=python_exec,
                target_agent_path=target_agent_path,
                dataset_dir=abs_dataset_dir,
                working_dir=gen_dir,
                stdout_log_file=stdout_log_file,
                config=env_config,
            )
        else:
            with open(stdout_log_file, "w", encoding="utf-8") as log_fh:
                process = subprocess.Popen(
                    [python_exec, "-u", target_agent_path, "--dataset_dir", abs_dataset_dir, "--working_dir", gen_dir],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                for line in process.stdout:
                    print(line, end="")
                    log_fh.write(line)
                return_code = process.wait()

        with open(stdout_log_file, encoding="utf-8") as f:
            target_agent_stdout = f.read()

        logger.info("=" * 60)

        if return_code != 0:
            target_agent_error_msg = f"Target agent failed with exit code {return_code}"
            logger.error(f"  ✗ Target agent execution failed with exit code {return_code}")
            logger.warning("  → Continuing with feedback agent despite target agent failure")
            return False, target_agent_stdout, target_agent_stderr, target_agent_error_msg
        else:
            logger.info("  ✓ Target agent execution completed successfully")
            return True, target_agent_stdout, target_agent_stderr, target_agent_error_msg

    except FileNotFoundError:
        logger.error(f"  ✗ Target agent file not found: {target_agent_path}")
        logger.error("  → Cannot continue.")
        return False, "", "", f"Target agent file not found: {target_agent_path}"
    except Exception as e:
        target_agent_error_msg = f"Unexpected error during target agent execution: {e!s}"
        logger.exception(f"  ✗ {target_agent_error_msg}")
        logger.warning("  → Continuing with feedback agent despite target agent failure")
        try:
            with open(stdout_log_file, encoding="utf-8") as f:
                target_agent_stdout = f.read()
        except OSError:
            pass
        return False, target_agent_stdout, target_agent_stderr, target_agent_error_msg


def _build_feedback_context(
    current_gen: int,
    gen_dir: str,
    dataset_dir: str,
    target_agent_success: bool,
    target_agent_error_msg: str,
    target_agent_stdout: str,
    target_agent_stderr: str,
    stdout_log_file: str,
    task_files: TaskFiles,
) -> tuple[str, str]:
    """Build execution status and section for feedback prompt.

    Returns (execution_status, execution_section).
    """
    # Load execution log
    agent_execution, is_multi_trajectory = load_agent_execution(gen_dir)

    if is_multi_trajectory:
        trajectory_count = agent_execution.get("count", 0)
        trajectories = agent_execution.get("trajectories", [])

        successful = sum(1 for t in trajectories if isinstance(t, list))
        failed = sum(1 for t in trajectories if isinstance(t, dict) and t.get("error"))

        sample_trajectories_text = ""
        for idx, traj in enumerate(trajectories[:3]):
            traj_json = json.dumps(traj, indent=2)
            if len(traj_json) > Config.TRAJECTORY_PREVIEW_LIMIT:
                traj_json = traj_json[: Config.TRAJECTORY_PREVIEW_LIMIT] + "\n  ... (truncated)"
            sample_trajectories_text += f"\n### Trajectory {idx}\n```json\n{traj_json}\n```\n"

        execution_section = f"""
**MULTI-TRAJECTORY EXECUTION**:

The agent executed {trajectory_count} separate trajectories (e.g., different questions/samples).

**Summary**:
- Total trajectories: {trajectory_count}
- Successful: {successful}
- Failed: {failed}
- Execution folder: {os.path.join(gen_dir, "agent_execution")}

**Sample Trajectories** (first 3 shown, you can read others from the folder):
{sample_trajectories_text}

**To analyze all trajectories**:
- Read files from: {os.path.join(gen_dir, "agent_execution")}
- Files named: execution_q0.json, execution_q1.json, ..., execution_q{trajectory_count - 1}.json

**Analysis guidance**:
- Look for common failure patterns across trajectories
- Check if trajectories are properly isolated
- Ensure consistent behavior across all samples
"""
    else:
        traj_json = json.dumps(agent_execution, indent=2)
        if len(traj_json) > Config.TRAJECTORY_PREVIEW_LIMIT:
            traj_json = traj_json[: Config.TRAJECTORY_PREVIEW_LIMIT] + "\n  ... (truncated)"
        execution_section = f"""
Here is the target agent execution trajectory:
```json
{traj_json}
```

NOTE: If you see an "error" field in the above JSON, it means the execution log was malformed or missing. Focus on making the agent more robust.
"""

    # Load evaluation results if available
    eval_results_section = ""
    results_json_path = os.path.join(gen_dir, "results.json")
    if os.path.exists(results_json_path):
        try:
            file_size = os.path.getsize(results_json_path)
            if file_size > Config.MAX_EXECUTION_LOG_SIZE:
                eval_results_section = f"\n**EVALUATION RESULTS**: results.json too large ({file_size:,} bytes)\n"
            else:
                with open(results_json_path, encoding="utf-8") as f:
                    eval_data = json.load(f)
                eval_results_section = f"""

**EVALUATION RESULTS**:
```json
{json.dumps(eval_data, indent=2)}
```
"""
        except (json.JSONDecodeError, OSError) as e:
            eval_results_section = f"\n**EVALUATION RESULTS**: Error loading results.json: {e}\n"
    else:
        eval_results_section = (
            "\n**EVALUATION RESULTS**: No results.json found (evaluation may not have run or may have failed)\n"
        )

    # Build execution status
    stdout_lines = target_agent_stdout.split("\n")
    last_10_lines = "\n".join(stdout_lines[-10:]) if len(stdout_lines) > 10 else target_agent_stdout

    if target_agent_success:
        execution_status = f"""SUCCESS: Target agent completed execution successfully.
{eval_results_section}

**Last 10 lines of output**:
```
{last_10_lines}
```

Full logs available at: {stdout_log_file}
"""
    else:
        execution_status = f"""FAILED: {target_agent_error_msg}
{eval_results_section}

**Last 10 lines of output**:
```
{last_10_lines}
```

Full logs available at: {stdout_log_file}

STDERR:
{target_agent_stderr}
"""

    return execution_status, execution_section


def _run_feedback_agent(
    current_gen: int,
    max_gen: int,
    run_dir: str,
    next_gen_dir: str,
    task_files: TaskFiles,
    execution_status: str,
    execution_section: str,
    meta_model: str,
    backend: AgentBackend,
    env_config: Config,
    dataset_dir: str,
    stdout_log_file: str,
) -> None:
    """Run the feedback agent to create an improved target agent."""
    agent_py = Path(os.path.join(run_dir, f"gen_{current_gen}"), "target_agent.py").read_text(encoding="utf-8")
    task = Path(dataset_dir, "task.md").read_text(encoding="utf-8")

    previous_gens_list = list(range(1, current_gen)) if current_gen > 1 else []
    previous_gens_text = ", ".join(map(str, previous_gens_list)) if previous_gens_list else "None"

    feedback_agent_prompt = build_feedback_prompt(
        current_gen=current_gen,
        max_gen=max_gen,
        task_files=task_files,
        agent_py=agent_py,
        task=task,
        execution_status=execution_status,
        execution_section=execution_section,
        run_dir=run_dir,
        next_gen_dir=next_gen_dir,
        previous_gens=previous_gens_text,
        stdout_log_file=stdout_log_file,
    )

    os.makedirs(next_gen_dir, exist_ok=True)

    feedback_prompt_path = os.path.join(next_gen_dir, "feedback_agent_prompt.txt")
    with open(feedback_prompt_path, "w", encoding="utf-8") as f:
        f.write(feedback_agent_prompt)
    logger.info(f"  ✓ Saved feedback agent prompt to: {feedback_prompt_path}")

    asyncio.run(
        run_agent(
            model_name=meta_model,
            max_turns=str(Config.DEFAULT_MAX_TURNS),
            prompt=feedback_agent_prompt,
            agent_working_directory=next_gen_dir,
            backend=backend,
        )
    )

    next_gen = current_gen + 1
    logger.info(f"Feedback agent completed. Created improved agent for generation {next_gen}")


def run_generation(
    current_gen: int,
    max_gen: int,
    run_setup: RunSetup,
    task_files: TaskFiles,
    abs_dataset_dir: str,
    dataset_dir: str,
    meta_model: str,
    backend: AgentBackend,
    sandbox: str,
    env_config: Config,
) -> None:
    """Execute one generation: run target agent, evaluate, optionally run feedback agent."""
    run_dir = run_setup.run_directory
    gen_dir = os.path.abspath(f"{run_dir}/gen_{current_gen}")
    target_agent_path = os.path.join(gen_dir, "target_agent.py")
    stdout_log_file = os.path.join(gen_dir, "target_agent_stdout.log")

    logger.info(f"Running target agent: {target_agent_path}")
    logger.info(f"  → Stdout log: {stdout_log_file}")
    logger.info("=" * 60)

    generation_start_time = time.time()

    # Run target agent
    target_agent_success, target_agent_stdout, target_agent_stderr, target_agent_error_msg = _run_target_agent(
        venv_dir=run_setup.venv_dir,
        target_agent_path=target_agent_path,
        abs_dataset_dir=abs_dataset_dir,
        gen_dir=gen_dir,
        stdout_log_file=stdout_log_file,
        sandbox=sandbox,
        env_config=env_config,
    )

    generation_duration = time.time() - generation_start_time

    # Run evaluation (if evaluate.py exists)
    logger.info("=" * 60)
    logger.info("Running evaluation (if available)...")
    run_evaluation(gen_dir, dataset_dir, run_setup.venv_dir)
    logger.info("=" * 60)

    # Add generation to context
    improvement_md_path = os.path.join(gen_dir, "improvement.md")
    run_setup.context_mgr.add_generation(
        gen_num=current_gen,
        gen_data={
            "success": target_agent_success,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration": generation_duration,
            "agent_path": target_agent_path,
            "gen_dir": gen_dir,
            "improvement_path": improvement_md_path if os.path.exists(improvement_md_path) else None,
            "execution_type": "Multi-trajectory"
            if (os.path.isdir(os.path.join(gen_dir, "agent_execution")))
            else "Single",
        },
    )

    # Run feedback agent (if not the last generation)
    if current_gen < max_gen:
        logger.info(f"Running feedback agent for generation {current_gen}")
        logger.info("Loading agent execution log...")

        execution_status, execution_section = _build_feedback_context(
            current_gen=current_gen,
            gen_dir=gen_dir,
            dataset_dir=dataset_dir,
            target_agent_success=target_agent_success,
            target_agent_error_msg=target_agent_error_msg,
            target_agent_stdout=target_agent_stdout,
            target_agent_stderr=target_agent_stderr,
            stdout_log_file=stdout_log_file,
            task_files=task_files,
        )

        next_gen = current_gen + 1
        next_gen_directory = os.path.abspath(f"{run_dir}/gen_{next_gen}")

        _run_feedback_agent(
            current_gen=current_gen,
            max_gen=max_gen,
            run_dir=run_dir,
            next_gen_dir=next_gen_directory,
            task_files=task_files,
            execution_status=execution_status,
            execution_section=execution_section,
            meta_model=meta_model,
            backend=backend,
            env_config=env_config,
            dataset_dir=dataset_dir,
            stdout_log_file=stdout_log_file,
        )
    else:
        logger.info(f"Generation {current_gen} is the final generation. Skipping feedback agent.")


def build_feedback_prompt(
    current_gen: int,
    max_gen: int,
    task_files: TaskFiles,
    agent_py: str,
    task: str,
    execution_status: str,
    execution_section: str,
    run_dir: str,
    next_gen_dir: str,
    previous_gens: str,
    stdout_log_file: str,
) -> str:
    """Build the feedback agent prompt for improving the target agent."""
    context_md_path = os.path.join(run_dir, "context.md")

    return f"""You are an expert AI Engineer analyzing agent scaffolds for iterative improvement.

**GENERATION CONTEXT**:
- Current generation: {current_gen}
- Previous generations: {previous_gens}
- Evolution history: {context_md_path}

**BEFORE ANALYZING - READ THE FULL HISTORY**:
1. Read {context_md_path} to understand:
   - What improvements were tried in each previous generation
   - Performance trends across generations
   - What worked and what didn't work
2. Review previous improvement.md files from earlier generations if helpful
3. Don't repeat failed approaches from earlier generations
4. Build upon successful patterns that improved performance

---

**SAMPLE TASK DESCRIPTIONS**:
```
{task_files.sample_task_descriptions}
```

**CURRENT TARGET AGENT** (Generation {current_gen}):
```python
{agent_py}
```

**TASK WORKED ON**:
```
{task}
```

**EXECUTION STATUS**:
```
{execution_status}
```

**EXECUTION LOGS**:
{execution_section}

---

**YOUR TASK**:

You must create exactly TWO files in {next_gen_dir}/:
1. improvement.md - Analysis and improvement plan
2. target_agent.py - The improved agent implementation

Follow these steps:

**STEP 1: Analyze the execution**:
   - For multi-trajectory: Look for patterns across all trajectories
   - For single-trajectory: Analyze the full execution flow
   - Identify what worked well and what failed
   - Check for consistency and robustness

**STEP 2: Review evolution history**:
   - Read context.md to see the full evolution
   - Understand what was tried in previous generations
   - Build upon successful patterns
   - Avoid repeating failed approaches

**STEP 3: Write improvement.md**:
   - MUST save to: {next_gen_dir}/improvement.md
   - Document your analysis and planned improvements
   - Focus on structural improvements to the agent scaffold
   - Make the agent more robust and generalizable
   - Don't optimize for this specific task
   - Reference insights from previous generations if applicable

**STEP 4: Create improved target_agent.py**:
   - MUST save to: {next_gen_dir}/target_agent.py
   - Implement the improvements documented in improvement.md
   - Apply all the planned improvements from step 3
   - Do not create or modify any other files besides these two

**RULES**:
- Focus on agent structure, not task-specific optimizations
- Make the agent work well across diverse task types (see sample task descriptions)
- If execution failed, fix the root cause
- If multi-trajectory: ensure each trajectory is properly isolated and logged
- Consider error handling, logging mechanisms, and robustness
- Build upon successful patterns from previous generations (check context.md)
- If execution log shows errors or is incomplete, suggest improvements to ensure proper logging

NOTE: The agent execution log may be incomplete or contain errors if the target agent crashed. If you see an "error" field, focus on making the agent more robust to prevent such failures.
"""


def main():
    _print_welcome()

    # Load env-var overrides (lower priority than explicit CLI flags)
    env_config = Config.from_env()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the orchestrator for agent evolution")
    parser.add_argument(
        "--max_gen",
        type=int,
        default=env_config.DEFAULT_MAX_GENERATIONS,
        help="Maximum number of generations to run (default: 3)",
    )
    parser.add_argument("--run_id", type=int, default=1, help="Run ID for this experiment (default: 1)")
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--task",
        type=str,
        choices=BUNDLED_TASKS,
        help=f"Name of a bundled task shipped with sia-agent ({', '.join(BUNDLED_TASKS)})",
    )
    task_group.add_argument(
        "--task_dir",
        type=str,
        help="Path to an external task directory (e.g., ./tasks/my-task)",
    )
    parser.add_argument(
        "--meta_model",
        type=str,
        default=None,
        help="Model to use for meta-agent (default: haiku for claude backend, gemini/gemini-3.1-pro-preview for openhands backend)",
    )
    parser.add_argument(
        "--task_model",
        type=str,
        default=env_config.DEFAULT_TASK_MODEL,
        help="Model to use for target agent (default: claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=env_config.DEFAULT_BACKEND,
        choices=["claude", "openhands"],
        help="Agent backend to use: claude (Claude Code SDK) or openhands (OpenHands SDK) (default: claude)",
    )
    parser.add_argument(
        "--sandbox",
        type=str,
        default=env_config.SANDBOX_MODE,
        choices=["none", "docker"],
        help="Sandbox mode for target agent execution: none (default) or docker (requires Docker)",
    )
    args = parser.parse_args()

    max_gen = args.max_gen
    task_dir, shared_dir = resolve_task_dir(args.task, args.task_dir)
    run_id = args.run_id
    backend = args.backend

    # Set default meta_model based on backend if not explicitly provided
    if args.meta_model is None:
        if backend == "openhands":
            meta_model = env_config.DEFAULT_OPENHANDS_META_MODEL
            logger.info(f"Using default OpenHands model: {env_config.DEFAULT_OPENHANDS_META_MODEL}")
        else:
            meta_model = env_config.DEFAULT_CLAUDE_META_MODEL
            logger.info(f"Using default Claude model: {env_config.DEFAULT_CLAUDE_META_MODEL}")
    else:
        meta_model = args.meta_model

    task_model = args.task_model

    logger.info("Configuration:")
    logger.info(f"  - Maximum generations: {max_gen}")
    logger.info(f"  - Task directory: {task_dir}")
    logger.info(f"  - Run ID: {run_id}")
    logger.info(f"  - Agent backend: {backend}")
    logger.info(f"  - Meta-agent model: {meta_model}")
    logger.info(f"  - Task-agent model: {task_model}")

    # ========================
    # SECTION 1: Load Files from Task Directory
    # ========================

    task_files = load_task_files(task_dir, shared_dir)

    # ========================
    # SECTION 2: Setup Run Directories
    # ========================

    run_setup = setup_run_directory(run_id, task_dir, meta_model, task_model, backend, max_gen)

    # ========================
    # SECTION 3: Build Initial Prompt
    # ========================

    meta_agent_prompt = build_meta_prompt(task_files, task_model, run_setup.meta_agent_working_directory)

    # ========================
    # SECTION 4: Run Target Agent Creation (Meta-Agent)
    # ========================

    # Save the meta-agent prompt for debugging/transparency
    meta_agent_prompt_path = os.path.join(run_setup.meta_agent_working_directory, "meta_agent_prompt.txt")
    with open(meta_agent_prompt_path, "w", encoding="utf-8") as f:
        f.write(meta_agent_prompt)
    logger.info(f"  ✓ Saved meta-agent prompt to: {meta_agent_prompt_path}")

    asyncio.run(
        run_agent(
            model_name=meta_model,
            max_turns=str(Config.DEFAULT_MAX_TURNS),
            prompt=meta_agent_prompt,
            agent_working_directory=run_setup.meta_agent_working_directory,
            backend=backend,
        )
    )

    # ========================
    # SECTION 5: Main Loop - Run Target Agent and Feedback Agent
    # ========================

    DATASET_DIRECTORY = os.path.join(task_dir, "data/public")
    ABS_DATASET_DIRECTORY = os.path.abspath(DATASET_DIRECTORY)
    logger.info(f"Dataset directory: {ABS_DATASET_DIRECTORY}")

    for current_gen in range(1, max_gen + 1):
        logger.info("=" * 80)
        logger.info(f"Starting Generation {current_gen} of {max_gen}")
        logger.info("=" * 80)

        run_generation(
            current_gen=current_gen,
            max_gen=max_gen,
            run_setup=run_setup,
            task_files=task_files,
            abs_dataset_dir=ABS_DATASET_DIRECTORY,
            dataset_dir=DATASET_DIRECTORY,
            meta_model=meta_model,
            backend=backend,
            sandbox=args.sandbox,
            env_config=env_config,
        )

    # Finalize context with summary statistics
    logger.info("Finalizing context.md with summary statistics...")
    run_setup.context_mgr.finalize()

    logger.info("=" * 80)
    logger.info(f"Orchestrator completed all {max_gen} generations successfully!")
    logger.info(f"Results saved in: {run_setup.run_directory}")
    logger.info(f"Context summary: {os.path.join(run_setup.run_directory, 'context.md')}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
