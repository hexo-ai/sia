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

import asyncio
import glob
import json
import os
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

from sia import __version__, cli
from sia.config import Config
from sia.harbor_runner import HarborRun, prepare_harbor_benchmark, run_generation_on_harbor
from sia.io_utils import file_size_ok, write_text
from sia.layout import BUNDLED_TASKS, Names, RunLayout, TaskLayout, resolve_task_dir, venv_python_path
from sia.logging_setup import configure_logging, get_logger
from sia.profiles import AgentProfile, load_profile
from sia.prompts import build_feedback_prompt, build_meta_prompt
from sia.results import FeedbackContext, TargetAgentResult
from sia.run_setup import RunSetup, TaskFiles, load_harbor_task_files, load_task_files, setup_run_directory
from sia.util import run_agent

__all__ = [
    "BUNDLED_TASKS",
    "RunSetup",
    "TaskFiles",
    "build_feedback_prompt",
    "build_meta_prompt",
    "load_agent_execution",
    "load_task_files",
    "main",
    "resolve_task_dir",
    "run_evaluation",
    "run_generation",
    "setup_run_directory",
]

logger = get_logger(__name__)


# ========================
# HELPER FUNCTIONS
# ========================


def load_agent_execution(gen_directory, config: Config | None = None):
    """
    Load execution logs with automatic format detection.

    Supports two formats:
    1. Single-file: gen_X/agent_execution.json (backwards compatible)
    2. Multi-trajectory: gen_X/agent_execution/execution_q0.json, execution_q1.json, ...

    Args:
        gen_directory: Path to the generation directory
        config: Optional Config instance (defaults to Config()).

    Returns:
        tuple: (execution_data, is_multi_trajectory)
            - execution_data: dict or list containing execution log(s)
            - is_multi_trajectory: bool indicating if multi-trajectory format
    """
    cfg = config or Config()
    execution_folder = os.path.join(gen_directory, Names.AGENT_EXECUTION_DIR)
    execution_file = os.path.join(gen_directory, Names.AGENT_EXECUTION_JSON)

    # Multi-trajectory folder: one file per question
    if os.path.isdir(execution_folder):
        logger.info("  → Detected multi-trajectory format (folder)")

        files = sorted(glob.glob(os.path.join(execution_folder, Names.EXECUTION_GLOB)))

        if not files:
            logger.warning("  ✗ agent_execution/ folder exists but is empty")
            return {"error": "Empty execution folder", "type": "multi-trajectory"}, True

        # Load all trajectory files
        trajectories = []
        for f in files:
            try:
                within_limit, file_size = file_size_ok(f, cfg.MAX_EXECUTION_LOG_SIZE)
                if not within_limit:
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

    # Single combined execution file
    elif os.path.exists(execution_file):
        logger.info("  → Detected single-file format")

        try:
            within_limit, file_size = file_size_ok(execution_file, cfg.MAX_EXECUTION_LOG_SIZE)
            if not within_limit:
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


def run_evaluation(gen_directory, task_dir, venv_dir, config: Config | None = None):
    """
    Run evaluate.py if it exists in the task's public data directory.

    Args:
        gen_directory: Path to the generation directory containing submission files
        task_dir: Path to the task directory
        venv_dir: Path to the virtual environment
        config: Optional Config instance (defaults to Config()).

    Returns:
        dict: Evaluation results or error information
    """
    cfg = config or Config()

    # Look for evaluate.py in data/public/ first, then fall back to task_dir
    evaluate_script = TaskLayout(task_dir, "").evaluate_script()

    if evaluate_script is None:
        logger.info(f"  → No evaluate.py found in {task_dir}, skipping evaluation")
        return {"status": "skipped", "reason": "evaluate.py not found"}

    logger.info(f"Running evaluation script: {evaluate_script}")

    # Create evaluation log file
    eval_log_file = os.path.join(gen_directory, Names.EVAL_LOG)
    logger.info(f"  → Evaluation log: {eval_log_file}")

    # Run evaluate.py as subprocess with --gen-dir
    try:
        python_exec = venv_python_path(venv_dir)
        result = subprocess.run(
            [python_exec, evaluate_script, "--gen-dir", gen_directory],
            capture_output=True,
            text=True,
            timeout=cfg.EVAL_TIMEOUT,
        )
        # Write combined output to log file
        eval_output = result.stdout + result.stderr
        write_text(eval_log_file, eval_output)

        if result.returncode != 0:
            logger.error(f"  ✗ Evaluation failed with exit code {result.returncode}")
            return {
                "status": "error",
                "reason": f"evaluate.py exited with code {result.returncode}",
                "log_path": eval_log_file,
                "output": eval_output,
            }

        # Check if results.json was created
        results_json_path = os.path.join(gen_directory, Names.RESULTS_JSON)
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
        logger.error(f"  ✗ Evaluation timed out after {cfg.EVAL_TIMEOUT}s")
        return {"status": "error", "reason": f"Evaluation timed out after {cfg.EVAL_TIMEOUT}s"}
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


def _stream_to_log(cmd: list[str], stdout_log_file: str) -> int:
    """Run ``cmd``, streaming merged stdout/stderr to the console and a log file.

    Returns the process exit code. This is the single place the target agent
    subprocess is launched; the Popen call stays in this module's namespace so it
    remains patchable in tests.
    """
    with open(stdout_log_file, "w", encoding="utf-8") as log_fh:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in process.stdout:
            print(line, end="")
            log_fh.write(line)
        return process.wait()


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

    return _stream_to_log(docker_cmd, stdout_log_file)


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
    python_exec = venv_python_path(venv_dir)

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
            cmd = [python_exec, "-u", target_agent_path, "--dataset_dir", abs_dataset_dir, "--working_dir", gen_dir]
            return_code = _stream_to_log(cmd, stdout_log_file)

        with open(stdout_log_file, encoding="utf-8") as f:
            stdout = f.read()

        logger.info("=" * 60)

        if return_code != 0:
            error_msg = f"Target agent failed with exit code {return_code}"
            logger.error(f"  ✗ Target agent execution failed with exit code {return_code}")
            logger.warning("  → Continuing with feedback agent despite target agent failure")
            return TargetAgentResult(False, stdout, "", error_msg).as_tuple()
        else:
            logger.info("  ✓ Target agent execution completed successfully")
            return TargetAgentResult(True, stdout, "", "").as_tuple()

    except FileNotFoundError:
        logger.error(f"  ✗ Target agent file not found: {target_agent_path}")
        logger.error("  → Cannot continue.")
        return TargetAgentResult(False, "", "", f"Target agent file not found: {target_agent_path}").as_tuple()
    except Exception as e:
        error_msg = f"Unexpected error during target agent execution: {e!s}"
        logger.exception(f"  ✗ {error_msg}")
        logger.warning("  → Continuing with feedback agent despite target agent failure")
        stdout = ""
        try:
            with open(stdout_log_file, encoding="utf-8") as f:
                stdout = f.read()
        except OSError:
            pass
        return TargetAgentResult(False, stdout, "", error_msg).as_tuple()


def _run_target_agent_on_harbor(target_agent_path: str, gen_dir: str, harbor: HarborRun) -> tuple[bool, str, str]:
    """Run this generation's agent on a Harbor benchmark. Returns (success, stdout, error_msg).

    run_generation_on_harbor writes results.json + agent_execution/ into gen_dir, so the
    downstream feedback path consumes them exactly as in local mode.
    """
    try:
        results = run_generation_on_harbor(
            target_agent_path=target_agent_path,
            gen_dir=gen_dir,
            task_model=harbor.task_model,
            benchmark=harbor.benchmark_path,
            working_dir=harbor.working_dir,
            include_tasks=harbor.include_tasks,
        )
    except Exception as e:
        msg = f"Harbor run failed: {e!s}"
        logger.exception(f"  ✗ {msg}")
        return False, msg, msg

    success = "error" not in results and results["n_tasks"] > 0
    summary = (
        f"Harbor benchmark: {results['benchmark']}\n"
        f"score (mean reward): {results['score']:.4f}\n"
        f"solved: {results['n_solved']}/{results['n_tasks']} | errors: {results['n_errors']}"
    )
    logger.info(f"  ✓ Harbor run complete | {summary.splitlines()[-1]}")
    return success, summary, results.get("error", "")


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
    config: Config | None = None,
) -> tuple[str, str]:
    """Build execution status and section for feedback prompt.

    Returns (execution_status, execution_section).
    """
    cfg = config or Config()

    # Load execution log
    agent_execution, is_multi_trajectory = load_agent_execution(gen_dir, config=cfg)

    if is_multi_trajectory:
        trajectory_count = agent_execution.get("count", 0)
        trajectories = agent_execution.get("trajectories", [])

        successful = sum(1 for t in trajectories if isinstance(t, list))
        failed = sum(1 for t in trajectories if isinstance(t, dict) and t.get("error"))

        sample_trajectories_text = ""
        for idx, traj in enumerate(trajectories[:3]):
            traj_json = json.dumps(traj, indent=2)
            if len(traj_json) > cfg.TRAJECTORY_PREVIEW_LIMIT:
                traj_json = traj_json[: cfg.TRAJECTORY_PREVIEW_LIMIT] + "\n  ... (truncated)"
            sample_trajectories_text += f"\n### Trajectory {idx}\n```json\n{traj_json}\n```\n"

        execution_section = f"""
**MULTI-TRAJECTORY EXECUTION**:

The agent executed {trajectory_count} separate trajectories (e.g., different questions/samples).

**Summary**:
- Total trajectories: {trajectory_count}
- Successful: {successful}
- Failed: {failed}
- Execution folder: {os.path.join(gen_dir, Names.AGENT_EXECUTION_DIR)}

**Sample Trajectories** (first 3 shown, you can read others from the folder):
{sample_trajectories_text}

**To analyze all trajectories**:
- Read files from: {os.path.join(gen_dir, Names.AGENT_EXECUTION_DIR)}
- Files named: execution_q0.json, execution_q1.json, ..., execution_q{trajectory_count - 1}.json

**Analysis guidance**:
- Look for common failure patterns across trajectories
- Check if trajectories are properly isolated
- Ensure consistent behavior across all samples
"""
    else:
        traj_json = json.dumps(agent_execution, indent=2)
        if len(traj_json) > cfg.TRAJECTORY_PREVIEW_LIMIT:
            traj_json = traj_json[: cfg.TRAJECTORY_PREVIEW_LIMIT] + "\n  ... (truncated)"
        execution_section = f"""
Here is the target agent execution trajectory:
```json
{traj_json}
```

NOTE: If you see an "error" field in the above JSON, it means the execution log was malformed or missing. Focus on making the agent more robust.
"""

    # Load evaluation results if available
    eval_results_section = ""
    results_json_path = os.path.join(gen_dir, Names.RESULTS_JSON)
    if os.path.exists(results_json_path):
        try:
            within_limit, file_size = file_size_ok(results_json_path, cfg.MAX_EXECUTION_LOG_SIZE)
            if not within_limit:
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

    return FeedbackContext(execution_status, execution_section).as_tuple()


def _run_feedback_agent(
    current_gen: int,
    max_gen: int,
    run_dir: str,
    next_gen_dir: str,
    task_files: TaskFiles,
    execution_status: str,
    execution_section: str,
    meta_profile: AgentProfile,
    env_config: Config,
    dataset_dir: str,
    stdout_log_file: str,
    task_text: str | None = None,
    harbor: bool = False,
) -> None:
    """Run the feedback agent to create an improved target agent."""
    agent_py = Path(os.path.join(run_dir, f"gen_{current_gen}"), Names.TARGET_AGENT).read_text(encoding="utf-8")
    task = task_text if task_text is not None else Path(dataset_dir, "task.md").read_text(encoding="utf-8")

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
        harbor=harbor,
    )

    os.makedirs(next_gen_dir, exist_ok=True)

    feedback_prompt_path = os.path.join(next_gen_dir, Names.FEEDBACK_PROMPT)
    write_text(feedback_prompt_path, feedback_agent_prompt)
    logger.info(f"  ✓ Saved feedback agent prompt to: {feedback_prompt_path}")

    asyncio.run(
        run_agent(
            model_name=meta_profile.model,
            max_turns=str(env_config.DEFAULT_MAX_TURNS),
            prompt=feedback_agent_prompt,
            agent_working_directory=next_gen_dir,
            backend=meta_profile.backend,
            provider=meta_profile.provider,
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
    meta_profile: AgentProfile,
    sandbox: str,
    env_config: Config,
    harbor: HarborRun | None = None,
) -> None:
    """Execute one generation: run target agent, evaluate, optionally run feedback agent.

    When ``harbor`` is set the generation is run on a Harbor benchmark (scored by the
    benchmark's own verifiers) instead of locally; the rest of the loop is unchanged.
    """
    run_dir = run_setup.run_directory
    layout = RunLayout(run_dir)
    gen_dir = layout.gen_dir(current_gen)
    target_agent_path = layout.target_agent(current_gen)
    stdout_log_file = layout.stdout_log(current_gen)

    logger.info(f"Running target agent: {target_agent_path}")
    logger.info(f"  → Stdout log: {stdout_log_file}")
    logger.info("=" * 60)

    generation_start_time = time.time()

    if harbor is not None:
        target_agent_success, target_agent_stdout, target_agent_error_msg = _run_target_agent_on_harbor(
            target_agent_path, gen_dir, harbor
        )
        target_agent_stderr = ""
        stdout_log_file = os.path.join(gen_dir, Names.HARBOR_RUN_LOG)
        generation_duration = time.time() - generation_start_time
    else:
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
        run_evaluation(gen_dir, dataset_dir, run_setup.venv_dir, config=env_config)
        logger.info("=" * 60)

    # Add generation to context
    improvement_md_path = layout.improvement_md(current_gen)
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
            if os.path.isdir(layout.agent_execution_dir(current_gen))
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
            config=env_config,
        )

        next_gen = current_gen + 1
        next_gen_directory = layout.gen_dir(next_gen)

        _run_feedback_agent(
            current_gen=current_gen,
            max_gen=max_gen,
            run_dir=run_dir,
            next_gen_dir=next_gen_directory,
            task_files=task_files,
            execution_status=execution_status,
            execution_section=execution_section,
            meta_profile=meta_profile,
            env_config=env_config,
            dataset_dir=dataset_dir,
            stdout_log_file=stdout_log_file,
            task_text=task_files.task_md if harbor is not None else None,
            harbor=harbor is not None,
        )
    else:
        logger.info(f"Generation {current_gen} is the final generation. Skipping feedback agent.")


def _run_web(args) -> None:
    """Dispatch for ``sia web``: serve the runs visualizer in the foreground."""
    configure_logging(args.log_level)
    from sia.web import serve

    serve(host=args.host, port=args.port, runs_dir=args.runs_dir, open_browser=not args.no_browser)


def main():
    configure_logging()
    _print_welcome()

    # Load env-var overrides (lower priority than explicit CLI flags)
    env_config = Config.from_env()

    # Parse command-line arguments
    args = cli.parse_args(env_config)

    if args.command == "web":
        _run_web(args)
        return

    # Apply CLI log level (overrides the import-time default / $SIA_LOG_LEVEL).
    configure_logging(args.log_level)

    # Start the live dashboard in a background thread so the run is watchable.
    if not args.no_web:
        from sia.web import serve_in_background

        serve_in_background(host=args.web_host, port=args.web_port, runs_dir=Names.RUNS_ROOT)

    max_gen = args.max_gen
    run_id = args.run_id

    # Resolve execution mode: exactly one of the four task selectors must be given.
    harbor_mode = bool(args.harbor_dataset or args.harbor_task_dir)
    n_selectors = sum(bool(x) for x in (args.task, args.task_dir, args.harbor_dataset, args.harbor_task_dir))
    if n_selectors != 1:
        raise SystemExit("Provide exactly one of --task, --task_dir, --harbor_dataset, or --harbor_task_dir")
    if harbor_mode:
        task_dir, shared_dir, benchmark = None, None, (args.harbor_dataset or args.harbor_task_dir)
    else:
        task_dir, shared_dir = resolve_task_dir(args.task, args.task_dir)
        benchmark = None

    # Resolve agent profiles (each bundles backend + model + provider).
    meta_profile = load_profile(args.meta_profile)
    target_profile = load_profile(args.target_profile)
    meta_model = meta_profile.model
    task_model = target_profile.model
    backend = meta_profile.backend
    target_provider = target_profile.provider

    logger.info("Configuration:")
    logger.info(f"  - Maximum generations: {max_gen}")
    if harbor_mode:
        logger.info(f"  - Harbor benchmark: {benchmark}")
        if args.harbor_include_task:
            logger.info(f"  - Harbor tasks (filter): {', '.join(args.harbor_include_task)}")
    else:
        logger.info(f"  - Task directory: {task_dir}")
    logger.info(f"  - Run ID: {run_id}")
    logger.info(
        f"  - Meta profile: {meta_profile.profile_id} (backend={backend}, model={meta_model}, "
        f"provider={meta_profile.provider.provider_id})"
    )
    logger.info(
        f"  - Target profile: {target_profile.profile_id} (model={task_model}, "
        f"provider={target_provider.provider_id}/{target_provider.client_kind})"
    )

    for label, prov in (("meta", meta_profile.provider), ("target", target_provider)):
        if not os.getenv(prov.api_key_env):
            logger.warning(f"  ⚠ {prov.api_key_env} is not set; the {label} agent may fail to authenticate.")

    # ========================
    # SECTION 1+2: Load Task Files + Setup Run Directories
    # ========================
    # Harbor mode needs the run directory before it can download the benchmark, and
    # runs the target agent in the benchmark's containers (no local venv).

    if harbor_mode:
        run_setup = setup_run_directory(
            run_id, benchmark, meta_model, task_model, backend, max_gen, config=env_config, create_venv=False
        )
        benchmark_path, sample_descriptions = prepare_harbor_benchmark(benchmark, run_setup.run_directory)
        task_files = load_harbor_task_files(benchmark, sample_descriptions)
        logger.info(f"  ✓ Harbor benchmark prepared at: {benchmark_path}")
    else:
        task_files = load_task_files(task_dir, shared_dir)
        run_setup = setup_run_directory(run_id, task_dir, meta_model, task_model, backend, max_gen, config=env_config)

    # ========================
    # SECTION 3: Build Initial Prompt
    # ========================

    meta_agent_prompt = build_meta_prompt(
        task_files, task_model, run_setup.meta_agent_working_directory, provider=target_provider, harbor=harbor_mode
    )

    # ========================
    # SECTION 4: Run Target Agent Creation (Meta-Agent)
    # ========================

    # Save the meta-agent prompt for debugging/transparency
    meta_agent_prompt_path = os.path.join(run_setup.meta_agent_working_directory, Names.META_PROMPT)
    write_text(meta_agent_prompt_path, meta_agent_prompt)
    logger.info(f"  ✓ Saved meta-agent prompt to: {meta_agent_prompt_path}")

    asyncio.run(
        run_agent(
            model_name=meta_model,
            max_turns=str(env_config.DEFAULT_MAX_TURNS),
            prompt=meta_agent_prompt,
            agent_working_directory=run_setup.meta_agent_working_directory,
            backend=backend,
            provider=meta_profile.provider,
        )
    )

    # ========================
    # SECTION 5: Main Loop - Run Target Agent and Feedback Agent
    # ========================

    if harbor_mode:
        dataset_directory = abs_dataset_directory = None
        harbor_run = HarborRun(
            benchmark_path=benchmark_path,
            task_model=task_model,
            working_dir=args.harbor_working_dir,
            include_tasks=args.harbor_include_task,
        )
    else:
        task_layout = TaskLayout(task_dir, shared_dir)
        dataset_directory = task_layout.dataset_dir
        abs_dataset_directory = task_layout.abs_dataset_dir
        harbor_run = None
        logger.info(f"Dataset directory: {abs_dataset_directory}")

    for current_gen in range(1, max_gen + 1):
        logger.info("=" * 80)
        logger.info(f"Starting Generation {current_gen} of {max_gen}")
        logger.info("=" * 80)

        run_generation(
            current_gen=current_gen,
            max_gen=max_gen,
            run_setup=run_setup,
            task_files=task_files,
            abs_dataset_dir=abs_dataset_directory,
            dataset_dir=dataset_directory,
            meta_profile=meta_profile,
            sandbox=args.sandbox,
            env_config=env_config,
            harbor=harbor_run,
        )

    # Finalize context with summary statistics
    logger.info("Finalizing context.md with summary statistics...")
    run_setup.context_mgr.finalize()

    logger.info("=" * 80)
    logger.info(f"Orchestrator completed all {max_gen} generations successfully!")
    logger.info(f"Results saved in: {run_setup.run_directory}")
    logger.info(f"Context summary: {os.path.join(run_setup.run_directory, Names.CONTEXT_MD)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
