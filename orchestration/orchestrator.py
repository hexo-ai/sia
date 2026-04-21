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
import sys
import json
import asyncio
import logging
import argparse
import time
from datetime import datetime

from util import run_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run the orchestrator for agent evolution')
parser.add_argument('--max_gen', type=int, default=3, help='Maximum number of generations to run (default: 3)')
parser.add_argument('--run_id', type=int, default=1, help='Run ID for this experiment (default: 1)')
parser.add_argument('--task_dir', type=str, required=True, help='Path to the task directory (e.g., ./tasks/task_1)')
parser.add_argument('--task_model', type=str, default='claude-haiku-4-5-20251001', help='Model for the target agent to use (default: haiku)')
args = parser.parse_args()

max_gen = args.max_gen
task_dir = args.task_dir
run_id = args.run_id
task_model = args.task_model

logger.info(f"Configuration:")
logger.info(f"  - Maximum generations: {max_gen}")
logger.info(f"  - Task directory: {task_dir}")
logger.info(f"  - Run ID: {run_id}")
logger.info(f"  - Task model: {task_model}")


# ========================
# SECTION 1: Load Files from Task Directory
# ========================

logger.info("Loading files from task directory...")

SAMPLE_TASK_DESCRIPTIONS = open(os.path.join(task_dir, "reference/SAMPLE_TASK_DESCRIPTIONS.md")).read()
logger.info("  ✓ Sample task descriptions loaded")

REFERENCE_TARGET_AGENT_PY = open(os.path.join(task_dir, "reference/reference_target_agent.py")).read()
logger.info("  ✓ Reference target agent loaded")

SAMPLE_AGENT_EXECUTION = json.load(open(os.path.join(task_dir, "../_shared/sample_agent_execution.json")))
logger.info("  ✓ Sample agent execution loaded")


# ========================
# SECTION 2: Setup Run Directories
# ========================

gen_num = 1
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
venv.create(venv_dir, with_pip=True)

# Path to the pip executable inside the venv
pip_executable = os.path.join(venv_dir, "bin", "pip")

# Install required packages: anthropic, python-dotenv
logger.info("Installing required packages: anthropic, python-dotenv in the virtual environment")
subprocess.run([pip_executable, "install", "anthropic", "python-dotenv"], check=True)


# ========================
# SECTION 3: Define Prompts
# ========================

META_AGENT_PROMPT = f"""You are a meta-agent. Your task is to create a target agent which can execute a task. Go ahead and create a target_agent.py for the target agent, which in turn can solve the given task.

Here are a couple of sample task descriptions which the target agent has to solve:
{SAMPLE_TASK_DESCRIPTIONS}

Here is a sample target_agent.py:
{REFERENCE_TARGET_AGENT_PY}

Here is a sample agent execution trajectory:
{json.dumps(SAMPLE_AGENT_EXECUTION, indent=2)}

CRITICAL RULES - FOLLOW EXACTLY:

1. The current working directory is {META_AGENT_WORKING_DIRECTORY}. Create the target_agent.py in the current working directory itself.

2. The target_agent.py MUST accept two command-line arguments:
   - --dataset_dir: Absolute path to the dataset directory (READ-ONLY, provided at runtime)
   - --working_dir: Absolute path to the working directory (READ-WRITE, provided at runtime)

3. CRITICAL: The target_agent.py must INCLUDE these paths in the prompt it sends to Claude. Claude MUST be explicitly told:
   - Where the dataset directory is located (the exact path from --dataset_dir)
   - Where the working directory is located (the exact path from --working_dir)
   - That it can ONLY READ from the dataset directory
   - That it can READ from and WRITE to the working directory

   DO NOT let Claude search for data in random locations. The prompt must say: "The dataset is at: <actual_dataset_dir_path>"

4. The target agent can ONLY read from the dataset directory provided via --dataset_dir, and can read from and write to the working directory specified by --working_dir. It must NOT access any other directories on the filesystem.
5. The target_agent.py should log its execution trajectory to a JSON file named 'agent_execution.json' in its working directory. This log should include all messages, tool calls, and their results. Use the same format as the sample agent execution trajectory provided above. Make sure to properly close the JSON file to avoid corruption.
6. Do NOT attempt to write to or modify files inside the dataset directory. It is READ-ONLY.
7. The target_agent.py should use only the "{task_model}" model when invoking the language model (do not use any other model).
8. DO NOT hardcode any specific dataset paths in the target_agent.py code. The paths will be provided at runtime via command-line arguments and MUST be passed to Claude in the prompt.

Example invocation (paths will vary at runtime):
    python target_agent.py --dataset_dir /path/to/dataset --working_dir /path/to/working
"""

FEEDBACK_AGENT_PROMPT = """You are an expert AI Engineer. Your task is to analyze an agent scaffold and its execution logs to suggest improvements to the scaffold.

Here are a couple of sample task descriptions which the agent is designed to solve.
```
{SAMPLE_TASK_DESCRIPTIONS}
```

Here is a target_agent.py which you created earlier
```
{AGENT_PY}
```

Here is the task which the target agent worked on:
```
{TASK}
```

Target Agent Execution Status:
```
{EXECUTION_STATUS}
```

Here is the target agent execution trajectory:
```
{AGENT_EXECUTION}
```

NOTE: The agent execution trajectory may be incomplete or contain errors if the target agent crashed or failed to complete its execution log properly. If you see an "error" field in the execution log, this indicates the log was malformed. In such cases, focus on improvements that would make the agent more robust and prevent such failures.

Your task is to analyze the target_agent.py and identify improvements that could be made to it.

RULES:
- Focus on the structure, approach, and methodology of the target agent itself
- Do NOT optimize for the specific task shown above
- Instead, think about how to make the target agent more robust and generalizable across the variety of tasks shown in the sample task descriptions
- Consider improvements to reasoning strategies, error handling, logging mechanisms, and overall agent architecture
- If the execution failed (see Execution Status above), analyze the error messages carefully and suggest specific fixes to prevent such failures
- If the execution log shows errors or is incomplete, suggest improvements to ensure the agent properly logs its execution and handles errors gracefully
- Provide thoughtful recommendations for enhancing the target agent's capabilities across diverse tasks.
- Consolidate all the improvements and write them to improvement.md in the current working directory: {IMPROVEMENT_DIR}
- Once you have written improvement.md, go ahead and implement the improvements ONLY in target_agent.py. Do not create or modify any other files.
"""


# ========================
# SECTION 4: Run Target Agent Creation (Meta-Agent)
# ========================

asyncio.run(run_agent(
    model_name="haiku",
    max_turns="20",
    prompt=META_AGENT_PROMPT,
    agent_working_directory=META_AGENT_WORKING_DIRECTORY
))


# ========================
# SECTION 5: Main Loop - Run Target Agent and Feedback Agent
# ========================

from pathlib import Path

# Define the dataset directory and working directory to pass as arguments
DATASET_DIRECTORY = os.path.join(task_dir, "data/public")
ABS_DATASET_DIRECTORY = os.path.abspath(DATASET_DIRECTORY)
logger.info(f"Dataset directory: {ABS_DATASET_DIRECTORY}")

# Run the loop for max_gen generations
# gen_1 is already created by meta-agent, so we loop from gen_1 to max_gen
for current_gen in range(1, max_gen + 1):
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
        # Build command with tee for real-time display and logging
        # Use PIPEFAIL to catch failures in the python command
        python_exec = os.path.join(venv_dir, "bin", "python")
        command = f"set -o pipefail; {python_exec} -u {target_agent_path} --dataset_dir {ABS_DATASET_DIRECTORY} --working_dir {current_gen_directory} 2>&1 | tee {stdout_log_file}"

        # Run with shell=True and bash to enable pipefail
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            executable='/bin/bash'  # Use bash to support pipefail
        )

        return_code = result.returncode

        # Read captured output from file for feedback agent
        with open(stdout_log_file, 'r') as f:
            target_agent_stdout = f.read()
        # Since we're using 2>&1, stderr is merged into stdout
        target_agent_stderr = ""

        logger.info(f"=" * 60)

        # Check if execution was successful
        if return_code != 0:
            target_agent_success = False
            target_agent_error_msg = f"Target agent failed with exit code {return_code}"
            logger.error(f"  ✗ Target agent execution failed with exit code {return_code}")
            logger.warning(f"  → Continuing with feedback agent despite target agent failure")
        else:
            target_agent_success = True
            logger.info(f"  ✓ Generation {current_gen} target agent execution completed successfully")

    except FileNotFoundError:
        logger.error(f"  ✗ Target agent file not found: {target_agent_path}")
        logger.error(f"  → Cannot continue. Exiting.")
        sys.exit(1)
    except Exception as e:
        target_agent_success = False
        target_agent_error_msg = f"Unexpected error during target agent execution: {str(e)}"
        logger.error(f"  ✗ {target_agent_error_msg}")
        logger.warning(f"  → Continuing with feedback agent despite target agent failure")

        # Try to read any partial logs
        try:
            with open(stdout_log_file, 'r') as f:
                target_agent_stdout = f.read()
        except:
            pass  # If log files don't exist, keep empty strings

    # Calculate execution duration
    generation_duration = time.time() - generation_start_time
    logger.info(f"  → Generation {current_gen} execution took {generation_duration:.2f} seconds")

    # ========================
    # SECTION 5b: Run Feedback Agent (if not the last generation)
    # ========================

    if current_gen < max_gen:
        logger.info(f"Running feedback agent for generation {current_gen}")

        # Load artifacts produced by the target agent so the feedback prompt is fully populated.
        AGENT_PY = Path(current_gen_directory, "target_agent.py").read_text(encoding="utf-8")
        TASK = Path(DATASET_DIRECTORY, "task.md").read_text(encoding="utf-8")

        # Try to load agent_execution.json with error handling
        agent_execution_path = Path(current_gen_directory, "agent_execution.json")
        try:
            with open(agent_execution_path, "r", encoding="utf-8") as f:
                AGENT_EXECUTION = json.load(f)
            AGENT_EXECUTION_PRETTY = json.dumps(AGENT_EXECUTION, indent=2)
            logger.info(f"  ✓ Successfully loaded agent execution log")
        except json.JSONDecodeError as e:
            logger.warning(f"  ✗ Failed to parse agent_execution.json: {e}")
            logger.warning(f"  → The target agent may have crashed or failed to complete the execution log")
            logger.warning(f"  → Using fallback execution log for feedback agent")

            # Read the raw content to see what we have
            try:
                with open(agent_execution_path, "r", encoding="utf-8") as f:
                    raw_content = f.read()
                logger.info(f"  → Raw file size: {len(raw_content)} bytes")

                # Create a fallback execution structure
                AGENT_EXECUTION = {
                    "error": "Failed to parse execution log",
                    "raw_content_preview": raw_content[:1000] if len(raw_content) > 1000 else raw_content,
                    "parse_error": str(e)
                }
                AGENT_EXECUTION_PRETTY = json.dumps(AGENT_EXECUTION, indent=2)
            except Exception as read_error:
                logger.error(f"  ✗ Could not read agent_execution.json at all: {read_error}")
                AGENT_EXECUTION = {"error": "Could not read execution log file"}
                AGENT_EXECUTION_PRETTY = json.dumps(AGENT_EXECUTION, indent=2)
        except FileNotFoundError:
            logger.error(f"  ✗ agent_execution.json not found at {agent_execution_path}")
            logger.error(f"  → The target agent did not create an execution log")
            AGENT_EXECUTION = {"error": "Execution log file not found"}
            AGENT_EXECUTION_PRETTY = json.dumps(AGENT_EXECUTION, indent=2)

        # Prepare execution status for feedback agent
        if target_agent_success:
            execution_status = "SUCCESS: Target agent completed execution successfully."
        else:
            execution_status = f"""FAILED: {target_agent_error_msg}

STDOUT:
{target_agent_stdout}

STDERR:
{target_agent_stderr}
"""

        # Prepare next generation directory
        next_gen = current_gen + 1
        next_gen_directory = os.path.abspath(f"{RUN_DIRECTORY}/gen_{next_gen}")

        # call feedback agent
        feedback_agent_prompt_prepared = FEEDBACK_AGENT_PROMPT.format(
            SAMPLE_TASK_DESCRIPTIONS=SAMPLE_TASK_DESCRIPTIONS,
            AGENT_PY=AGENT_PY,
            TASK=TASK,
            EXECUTION_STATUS=execution_status,
            AGENT_EXECUTION=AGENT_EXECUTION_PRETTY,
            IMPROVEMENT_DIR=next_gen_directory,
        )

        os.makedirs(next_gen_directory, exist_ok=True)
        asyncio.run(
            run_agent(
                model_name="haiku",
                max_turns="20",
                prompt=feedback_agent_prompt_prepared,
                agent_working_directory=next_gen_directory,
            )
        )

        logger.info(f"Feedback agent completed. Created improved agent for generation {next_gen}")
    else:
        logger.info(f"Generation {current_gen} is the final generation. Skipping feedback agent.")

logger.info(f"=" * 80)
logger.info(f"Orchestrator completed all {max_gen} generations successfully!")
logger.info(f"Results saved in: {RUN_DIRECTORY}")
logger.info(f"=" * 80)
