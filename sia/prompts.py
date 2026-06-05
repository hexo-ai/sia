"""Prompt builders for the meta-agent and feedback-agent.

Moved verbatim out of orchestrator.py. The exact text is product-critical and is
locked by the golden-master tests in tests/test_prompts_snapshot.py.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sia.providers import Provider
    from sia.run_setup import TaskFiles


# Appended ONLY in Harbor mode. The base prompts above stay byte-identical (golden-master
# safe); this block redefines the target agent's contract for in-container execution so it
# satisfies a Harbor task's native verifier instead of writing a local submission file.
HARBOR_META_PROMPT = """

================================================================================
HARBOR MODE — IN-CONTAINER EXECUTION CONTRACT (OVERRIDES THE DATASET/SUBMISSION RULES ABOVE)
================================================================================
Your target_agent.py is NOT run against a local dataset directory and there is NO
submission file or evaluate.py. Instead it is uploaded into the Docker container of
each benchmark task and executed INSIDE that container; the task's own verifier
inspects the final state of the container to score you.

Use THIS command-line contract (ignore the earlier --dataset_dir / submission rules):

    python3 target_agent.py --working_dir <dir> --instruction_file <path> --log_dir <dir>

  --working_dir       directory to operate in; create the required output files here
  --instruction_file  path to the task instruction describing what to accomplish
  --log_dir           write your trajectory to <log_dir>/agent_execution.json

Runtime guarantees inside the container:
  * internet is available,
  * an LLM API key is exported (ANTHROPIC_API_KEY),
  * the solver model id is in env var SIA_TASK_MODEL (use it; it equals {task_model}),
  * the `anthropic` SDK is installed.

Build an agentic loop: read the instruction, use the model with a bash tool to explore
the working directory, create/edit files, and verify your work, then stop when done.
The "reference target agent" shown above already follows THIS exact contract — model
your target_agent.py on it. Record every step to <log_dir>/agent_execution.json. Do NOT
write a submission file; satisfy the task by leaving the container in the requested state.
"""

HARBOR_FEEDBACK_PROMPT = """

================================================================================
HARBOR MODE
================================================================================
The target agent runs INSIDE each benchmark task's Docker container (no dataset dir,
no submission file). Its CLI contract is:
    python3 target_agent.py --working_dir <dir> --instruction_file <path> --log_dir <dir>
The score is the mean reward from the benchmark's own verifiers — see the EVALUATION
RESULTS (results.json): per_task rewards, n_solved, n_errors. When you rewrite
target_agent.py, KEEP this exact CLI contract and the in-container agentic-loop pattern;
focus on making the agent explore, edit files, and verify more reliably across diverse
tasks. Do NOT introduce a submission-file workflow.
"""


def build_meta_prompt(
    task_files: TaskFiles,
    task_model: str,
    working_dir: str,
    provider: Provider | None = None,
    harbor: bool = False,
) -> str:
    """Build the meta-agent prompt for creating the initial target agent.

    For Anthropic and Google providers (and the default ``None``) the text is
    byte-identical to the original. For OpenAI-compatible providers a client-setup
    block is prepended instructing the meta-agent to refactor the reference agent to
    the ``openai`` SDK at the provider's base_url/api_key_env.
    """
    base = f"""You are a meta-agent. Your task is to create a target agent which can execute a task. Go ahead and create a target_agent.py for the target agent, which in turn can solve the given task.

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
    prompt = base if provider is None or provider.client_kind != "openai" else build_target_client_setup(provider, task_model) + base
    if harbor:
        prompt = prompt + HARBOR_META_PROMPT.replace("{task_model}", task_model)
    return prompt


def build_target_client_setup(provider: Provider, task_model: str) -> str:
    """Prompt block telling the meta-agent how to reach an OpenAI-compatible target model.

    The reference target_agent.py shown later in the prompt may use a different SDK
    (e.g. the Gemini SDK); this block instructs the meta-agent to refactor it to the
    ``openai`` SDK configured for ``provider``.
    """
    return f"""=== TARGET MODEL CLIENT SETUP (OpenAI-compatible provider: {provider.name}) ===

The target model "{task_model}" is served by an OpenAI-compatible API. The reference
target_agent.py shown below may use a different SDK (e.g. the Gemini SDK) — you MUST
refactor your target_agent.py to use the `openai` SDK configured for this provider
(do NOT use the anthropic or google SDK):

    import os
    from openai import OpenAI

    client = OpenAI(
        base_url="{provider.base_url}",
        api_key=os.environ["{provider.api_key_env}"],
    )

Call client.chat.completions.create(model="{task_model}", ...) using OpenAI-style
messages (and OpenAI function calling / response_format where the reference uses
structured output). Do NOT compute a dollar cost: per-provider pricing is unknown, so
set any cost field to 0 (token counts from the API response are still fine to record).

"""


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
    harbor: bool = False,
) -> str:
    """Build the feedback agent prompt for improving the target agent."""
    context_md_path = os.path.join(run_dir, "context.md")

    prompt = f"""You are an expert AI Engineer analyzing agent scaffolds for iterative improvement.

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
    if harbor:
        prompt = prompt + HARBOR_FEEDBACK_PROMPT
    return prompt
