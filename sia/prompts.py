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

# Generic, task-agnostic instruction injected into the meta- and feedback-agent
# harness-mode prompts. It tells the agent that held-out grader ground truth exists
# in a private directory and must never be accessed — without naming any concrete
# path, so it works for any SIA task.
HELD_OUT_GROUND_TRUTH_NOTICE = (
    "HELD-OUT EVALUATION INTEGRITY: The task's held-out evaluation ground truth lives in a "
    "private directory used only by the grader. You MUST NOT read, list, glob, or otherwise "
    "access it — improve solely from the feedback provided. Such access is blocked and would "
    "invalidate the experiment."
)


def _reference_section(task_files: TaskFiles, reference_dir: str | None) -> str:
    """The reference paragraph of the meta prompt.

    Default/single-file: embed the seed code verbatim (byte-identical to the original).
    Multi-file directory: point the agent at the on-disk reference and tell it to read
    the files itself and declare any extra deps in requirements.txt.
    """
    if reference_dir is None:
        return (
            "Here is a sample target_agent.py showing the complete implementation pattern "
            f"(READ THE ENTIRE FILE):\n{task_files.reference_target_agent_py}"
        )
    return (
        f"Your reference agent implementation has been placed in your working directory ({reference_dir}). "
        "It may span multiple files. READ IT YOURSELF with your file tools (Read/Glob/Grep) — study the "
        "entrypoint and any helper modules — then write your target_agent.py in the same directory.\n"
        "If your target_agent.py needs third-party packages, list them (one per line) in a requirements.txt "
        "in your working directory; they are installed before the target agent runs."
    )


def _build_weights_meta_prompt(
    task_files: TaskFiles,
    task_model: str,
    working_dir: str,
    training_sandbox: str = "modal",
) -> str:
    """Build the meta-agent prompt for RL-based weight tuning (train.py).

    Args:
        training_sandbox: "modal" (default) or "sandboxfusion" for code execution
    """
    # RL Integration Guide (sections 1-9)
    RL_GUIDE = """# RL Integration Guide: Custom Task Tuning with Tinker-Cookbook

This guide provides the necessary architectural context and implementation patterns to integrate a custom Agent/Task into the `tinker-cookbook` RL pipeline. Use this to build a task-specific `Env`, `EnvGroupBuilder`, and `RLDataset`.

---

## 1. Core Abstractions Mapping

| Tinker Class | Role in RL | Custom Implementation Goal |
| :--- | :--- | :--- |
| **`Env`** | The "World" | Manages a single trajectory (one agent, one problem). Handles tool-calls and intermediate rewards. |
| **`EnvGroupBuilder`** | The "Orchestrator" | Creates $N$ environments for the **same** problem. Handles final group-level rewards (GRPO). |
| **`RLDataset`** | The "Task Source" | Groups `EnvGroupBuilder` instances into batches. Feeds the training loop. |

---

## 2. Implementing the `Env`

The `Env` must manage the conversation state and define how the agent interacts with tools.

```python
from tinker_cookbook.rl.types import Env, StepResult
from tinker_cookbook.renderers import Renderer

class CustomAgentEnv(Env):
    def __init__(self, task_data, renderer: Renderer):
        self.task = task_data
        self.renderer = renderer
        self.messages = []

    async def initial_observation(self):
        self.messages = [{"role": "user", "content": self.task["query"]}]
        prompt, stop_cond = self.renderer.build_generation_prompt(self.messages)
        return prompt, stop_cond

    async def step(self, action_tokens: list[int], extra=None):
        response_text = self.renderer.tokenizer.decode(action_tokens)

        # LOGIC: Execute tools or parse final answer
        tool_result, is_done = await execute_agent_tools(response_text)

        # Step reward (e.g., formatting check)
        reward = -0.1 if "invalid_tool" in response_text else 0.0

        self.messages.append({"role": "assistant", "content": response_text})
        if tool_result:
            self.messages.append({"role": "user", "content": f"Result: {tool_result}"})

        next_obs, next_stop = self.renderer.build_generation_prompt(self.messages)
        return StepResult(
            reward=reward, episode_done=is_done,
            next_observation=next_obs, next_stop_condition=next_stop
        )
```

---

## 3. Implementing the `EnvGroupBuilder`

This class is responsible for spawning $N$ environments for a single problem and computing the final outcome.

```python
from tinker_cookbook.rl.types import EnvGroupBuilder, Trajectory
from typing import Sequence

class CustomTaskGroupBuilder(EnvGroupBuilder):
    def __init__(self, task_data, group_size: int, renderer: Renderer):
        self.task = task_data
        self.group_size = group_size
        self.renderer = renderer

    async def make_envs(self) -> Sequence[CustomAgentEnv]:
        # Return N copies of the environment for the same task
        return [CustomAgentEnv(self.task, self.renderer) for _ in range(self.group_size)]

    async def compute_group_rewards(self, trajectories: list[Trajectory], envs: Sequence[CustomAgentEnv]):
        # Called after all N rollouts finish. Use this for "Final Success" rewards.
        rewards_and_metrics = []
        for traj in trajectories:
            final_text = self.renderer.tokenizer.decode(traj.transitions[-1].ac.tokens)
            is_correct = check_success(final_text, self.task["answer"])

            # Final Reward: 1.0 for success, 0.0 for failure
            reward = 1.0 if is_correct else 0.0
            rewards_and_metrics.append((reward, {"is_correct": float(is_correct)}))

        return rewards_and_metrics
```

---

## 5. Implementing a Custom `RLDatasetBuilder`

The `RLDatasetBuilder` is the entry point for your data. It must be a `@chz.chz` dataclass.

```python
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
from tinker_cookbook import renderers, tokenizer_utils
import chz
import json

class CustomTaskDataset(RLDataset):
    def __init__(self, tasks, batch_size: int, group_size: int, renderer):
        self.builders = [CustomTaskGroupBuilder(t, group_size, renderer) for t in tasks]
        self.batch_size = batch_size

    def get_batch(self, index: int):
        start = index * self.batch_size
        return self.builders[start : start + self.batch_size]

    def __len__(self):
        return (len(self.builders) + self.batch_size - 1) // self.batch_size

@chz.chz
class MyDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    group_size: int
    model_name: str
    renderer_name: str = "qwen3_instruct"
    data_path: str = "data.jsonl"

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        # 1. Load your raw data
        with open(self.data_path) as f:
            tasks = [json.loads(line) for line in f]

        # 2. Setup Renderer & Tokenizer
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        # 3. Return Train (and optional Test) Dataset
        train_ds = CustomTaskDataset(tasks, self.batch_size, self.group_size, renderer)
        return train_ds, None
```

### ⚠️ Critical Warning on Missing Ground Truth, Reward Design, & Data Leakage ⚠️

When dynamically parsing task datasets (e.g., in your `RLDatasetBuilder`), be highly aware of the following realities regarding dataset structure, missing answers, and evaluation constraints:

1. **Dataset Paths:** The provided `--dataset_dir` will point directly to the accessible dataset. You must train using only this accessible data. Be robust in your file discovery (e.g., using `glob` to find JSON/CSV files and `evaluate.py`).
2. **Missing Ground Truth:** The available dataset is typically designed for evaluation and **may have its ground truth answers hidden or completely missing**. If you naively parse these files and grade against non-existent answers, your group rewards will be a constant `0.0` or `0.5`, leading to zero variance and failed GRPO training.
3. **Reward Workarounds (Crucial):** Because the ground truth answers may be missing, **you must implement a workaround to compute meaningful rewards**. A highly recommended approach is **Majority Voting (Self-Consistency)** across the generated samples within the `EnvGroupBuilder`. By rewarding the most common answer in the group, you provide a non-constant reward signal (variance) without needing the actual answer key. Do not let the reward remain constant.
4. **Data Leakage & Shuffling:** Make sure you properly shuffle the train set. If you do have access to ground truth, ensure you properly split your dataset into train and test sets to prevent overfitting. Do not evaluate your final inference run on the exact same examples you used for training.

---

## 6. Task-Specific Grading & Reward Computation

**Each task has its own `evaluate.py`** that defines how to grade agent outputs. Your `Env` and `EnvGroupBuilder` must integrate with this task-specific logic.

### Grading Examples by Task Type

- **GPQA**: Compare predicted answer letters (A/B/C/D) to ground truth
- **LawBench**: Compare predicted charge labels to ground truth categories
- **Chess**: Compare predicted moves to best moves in ground truth
- **Code Tasks**: Execute generated code and check against test cases

Your `EnvGroupBuilder.compute_group_rewards()` should:
1. Get the agent's final output
2. Call the task's grading logic (via `evaluate.py` or imported functions)
3. Convert results to rewards (e.g., `1.0` if correct, `0.0` if wrong)
4. Return rewards + metrics

```python
async def compute_group_rewards(self, trajectories, envs):
    \"\"\"Compute final rewards using task-specific grading.\"\"\"
    rewards_and_metrics = []

    for traj in trajectories:
        # 1. Extract final output from trajectory
        final_output = extract_output(traj)

        # 2. Call task-specific grading function
        # (imported from task's evaluate.py or grading module)
        is_correct, metrics = await grade_task(final_output, self.task)

        # 3. Convert to reward
        reward = 1.0 if is_correct else 0.0
        rewards_and_metrics.append((reward, metrics))

    return rewards_and_metrics
```

---

## 7. Reward Shaping Strategy

Tinker-cookbook uses **GRPO** (Group Relative Policy Optimization).

### Step Rewards (During Trajectory - `Env.step`)

**Format Correctness** is critical:

```python
async def step(self, action_tokens: list[int], extra=None):
    response_text = self.renderer.tokenizer.decode(action_tokens)

    # Check FORMAT correctness (task-specific)
    format_reward = 0.0

    # GPQA: Must output JSON with answer field
    if self.task_type == "gpqa":
        if is_valid_json_answer(response_text):  # {"answer": "A"}
            format_reward = +0.5  # Reward correct format
        else:
            format_reward = -0.5  # Penalize wrong format

    # LawBench: Must extract to valid charge label
    elif self.task_type == "lawbench":
        if can_extract_charge_label(response_text):  # Must match one of 191 classes
            format_reward = +0.5
        else:
            format_reward = -0.5

    # Chess: Must output "solution = [...] or solution = <number>"
    elif self.task_type == "chess":
        if has_solution_field(response_text):  # solution = ...
            format_reward = +0.5
        else:
            format_reward = -0.5

    # Other step rewards
    per_turn_penalty = -0.01  # Penalize long responses
    hallucination_penalty = -0.3 if "hallucinate" in response_text else 0.0

    # Total step reward
    reward = format_reward + per_turn_penalty + hallucination_penalty

    return StepResult(reward=reward, ...)
```

### Final Rewards (After Episode - `EnvGroupBuilder.compute_group_rewards`)

```python
async def compute_group_rewards(self, trajectories, envs):
    \"\"\"Final rewards based on correctness via task grading.\"\"\"
    rewards_and_metrics = []

    for traj in trajectories:
        final_output = extract_output(traj)

        # Call task-specific grader
        is_correct, metrics = await grade_task(final_output, self.task)

        # Final reward: 1.0 for correct, 0.0 for wrong
        final_reward = 1.0 if is_correct else 0.0

        rewards_and_metrics.append((final_reward, metrics))

    return rewards_and_metrics
```

### Overall Reward Strategy

| Type | Where | Examples | Purpose |
|------|-------|----------|---------|
| **Format Reward** | `Env.step()` | +0.5 correct, -0.5 wrong | Guide model to output correct format early |
| **Per-turn Penalty** | `Env.step()` | -0.01/step | Encourage conciseness |
| **Correctness Reward** | `EnvGroupBuilder.compute_group_rewards()` | +1.0 correct, 0.0 wrong | Final success metric |

### General Rules

1.  **Don't Normalize:** Tinker automatically centers rewards per group (`advantage = reward - group_mean`).
2.  **Format is Critical:** Wrong format = model can't be graded = negative step reward.
3.  **Correctness is Final:** Use `EnvGroupBuilder.compute_group_rewards` for final outcome-based rewards.
4.  **Sparse is Fine:** RL works best when the model can clearly distinguish a "Winner" from a "Loser" within the same 8-sample group.

---

## 8. The Training Pipeline

```python
from tinker_cookbook.rl.train import main, Config
import asyncio

async def run():
    config = Config(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        recipe_name="custom_task_tuning",
        dataset_builder=MyDatasetBuilder(
            batch_size=16,
            group_size=8,
            model_name="Qwen/Qwen3-4B-Instruct-2507"
        ),
        learning_rate=1e-5,
        max_tokens=1024,
        log_path="./runs/experiment_1"
    )
    await main(config)

if __name__ == "__main__":
    asyncio.run(run())
```

---

## 9. Troubleshooting: Zero Scores

If your training loop shows **score = 0** across all trajectories, follow these debugging steps:

### Step 1: Verify Dataset Alignment
- **Issue:** You may be scoring predictions for answers that don't exist in your dataset
- **Fix:**
  - Check that `compute_group_rewards()` is comparing against the correct answer field in `self.task`
  - Verify the answer exists and is not empty or malformed
  - Print the final answer and expected answer for the first few samples to confirm alignment

### Step 2: Check the Sandbox (if using code execution)
If your agent executes code via a sandbox:

#### If using **Local Sandbox (SandboxFusion)**:
1. **Verify it's running:**
   ```bash
   curl http://localhost:8080/health
   ```
   Should return a 200 response.

2. **Check concurrency limits:**
   - If concurrency is maxed out, the machine will freeze
   - Ensure you're not running too many concurrent requests
   - Add reasonable timeout and retry logic:
   ```python
   client = SandboxFusionClient(
       base_url="http://localhost:8080",
       timeout=30,
       max_retries=2
   )
   ```

3. **Test with a sample script:**
   ```python
   # Test if sandbox is working and returning expected results
   test_code = \"\"\"
   print("Hello from sandbox")
   result = 1 + 1
   print(f"Result: {result}")
   \"\"\"
   result = await client.run(
       code=test_code,
       timeout=10
   )
   print(result.stdout)  # Should show "Hello from sandbox" and "Result: 2"
   ```

#### If using **Modal (Cloud Sandbox)**:
- Modal should handle concurrency automatically
- If scores are still 0, verify your `modal.Image` has all required dependencies installed
- Check Modal logs for execution errors

### Step 3: Isolate the Problem
- **If sandbox test passes:** The problem is in your `Env` or reward logic. Check `compute_group_rewards()`
- **If sandbox test fails:** Fix sandbox connectivity before proceeding with training"""

    # Configure sandbox instruction based on training_sandbox choice
    if training_sandbox == "sandboxfusion":
        sandbox_instruction = """10. SANDBOX FUSION URL (IMPORTANT FOR CODE EXECUTION):
    - If your train.py uses SandboxFusionClient for code execution, configure it with:
      ```python
      import os
      from tinker_cookbook.sandbox import SandboxFusionClient
      sandbox_url = os.getenv("SANDBOX_URL", "http://localhost:8080")
      client = SandboxFusionClient(base_url=sandbox_url)
      ```
    - The SANDBOX_URL environment variable will be passed automatically at runtime"""
    else:
        sandbox_instruction = """10. MODAL CONFIGURATION (IMPORTANT FOR CODE EXECUTION):
    - Use Modal SDK (https://modal.com) as the default sandbox for executing code in your RL pipeline
    - For code execution in your RL components, use Modal Functions with @modal.function decorator
    - Modal provides better reliability, isolation, and scaling compared to alternatives
    - Example:
      ```python
      import modal
      app = modal.App("training-app")

      @app.function()
      def execute_code(code: str) -> str:
          exec(code)
          return result
      ```
    - Ensure MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are set in the environment (automatically available)"""

    return f"""You are a meta-agent. Your task is to create a training script which can execute a task.
In this 'weights' mode, your primary goal is to implement a Reinforcement Learning (RL) pipeline to tune the model's performance for the specific task using the `tinker-cookbook` library.

---
RL INTEGRATION GUIDE:
{RL_GUIDE}
---

Go ahead and create a train.py which will train and return back the training model checkpoint url. You should take the reference agent and tune the model using this agent and whatever data or problem statement we have.

Here is the FULL TASK SPECIFICATION that your train.py will need to solve:
{task_files.task_md}

Here are a couple of sample task descriptions which the reference agent has to solve:
{task_files.sample_task_descriptions}

Here is a sample reference train.py implementation showing the training component patterns (READ THE ENTIRE FILE):
{task_files.reference_target_agent_py}

Here is a sample agent execution trajectory:
{json.dumps(task_files.sample_agent_execution, indent=2)}

CRITICAL RULES - FOLLOW EXACTLY:

1. The current working directory is {working_dir}. Create the train.py in the current working directory itself.

2. The train.py MUST accept two command-line arguments:
   - --dataset_dir: Absolute path to the dataset directory (READ-ONLY, provided at runtime)
   - --working_dir: Absolute path to the working directory (READ-WRITE, provided at runtime)

3. CRITICAL: The train.py must INCLUDE these paths in the prompt it sends to {task_model}. {task_model} MUST be explicitly told:
   - Where the dataset directory is located (the exact path from --dataset_dir)
   - Where the working directory is located (the exact path from --working_dir)
   - That it can ONLY READ from the dataset directory
   - That it can READ from and WRITE to the working directory

   DO NOT let {task_model} search for data in random locations. The prompt must say: "The dataset is at: <actual_dataset_dir_path>"

4. The train.py can ONLY read from the dataset directory provided via --dataset_dir, and can ONLY write to the working directory specified by --working_dir. It must NOT access any other directories on the filesystem.

5. EXECUTION LOGGING - CRITICAL: The train.py must log its execution trajectory in agent_execution.json in the working directory. Include all messages, tool calls, and their results in valid JSON format.

6. Do NOT attempt to write to or modify files inside the dataset directory. It is READ-ONLY.
7. The train.py should use only the "{task_model}" model when invoking the language model (do not use any other model).
8. DO NOT hardcode any specific dataset paths in the train.py code. The paths will be provided at runtime via command-line arguments and MUST be passed to {task_model} in the prompt.

9. CRITICAL RULES FOR WEIGHTS MODE - FOLLOW EXACTLY:

   **MANDATORY:** Your train.py must implement only tinker-cookbook components (Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder). Do NOT write local training code or custom training loops. The tinker-cookbook library will execute your components directly via tinker_cookbook.rl.train.main().

   Your train.py must have this exact structure (copy-paste the template below):

   ```python
   import asyncio
   import os
   import sys
   import json
   from pathlib import Path
   from tinker_cookbook.rl.train import main as tinker_train_main, Config
   from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
   import chz

   # 1. Implement Env class (manages single trajectory)
   class MyEnv(Env):
       def __init__(self, task_data):
           self.task = task_data
           # ... implement __init__, initial_observation(), step()

       async def initial_observation(self):
           # Return initial observation
           pass

       async def step(self, action_tokens, extra=None):
           # Execute step and return StepResult
           pass

   # 2. Implement EnvGroupBuilder class (creates N environments, computes rewards)
   class MyGroupBuilder(EnvGroupBuilder):
       def __init__(self, task_data, group_size):
           self.task = task_data
           self.group_size = group_size

       async def make_envs(self):
           return [MyEnv(self.task) for _ in range(self.group_size)]

       async def compute_group_rewards(self, trajectories, envs):
           # Compute rewards using task-specific grading
           rewards_and_metrics = []
           for traj in trajectories:
               # Extract final output, grade it, compute reward
               reward = 1.0 if correct else 0.0
               rewards_and_metrics.append((reward, {{"is_correct": float(correct)}}))
           return rewards_and_metrics

   # 3. Implement RLDataset class (batches EnvGroupBuilders)
   class MyRLDataset(RLDataset):
       def __init__(self, tasks, batch_size, group_size):
           self.builders = [MyGroupBuilder(t, group_size) for t in tasks]
           self.batch_size = batch_size

       def get_train_batch(self, idx):
           start = idx * self.batch_size
           return self.builders[start:start + self.batch_size]

       def num_train_batches(self):
           return (len(self.builders) + self.batch_size - 1) // self.batch_size

       def get_eval_batch(self, idx):
           return self.get_train_batch(idx)

       def num_eval_batches(self):
           return self.num_train_batches()

   # 4. Implement RLDatasetBuilder class with @chz.chz decorator
   @chz.chz
   class MyDatasetBuilder(RLDatasetBuilder):
       batch_size: int
       group_size: int
       model_name: str

       async def __call__(self):
           tasks = load_tasks_from_dataset_dir()  # Load from --dataset_dir
           dataset = MyRLDataset(tasks, self.batch_size, self.group_size)
           return dataset, None

   # 5. Main function - call tinker_cookbook.rl.train.main()
   async def main(dataset_dir, working_dir):
       config = Config(
           model_name="Qwen/Qwen3-4B-Instruct-2507",
           recipe_name="custom_task_tuning",
           dataset_builder=MyDatasetBuilder(
               batch_size=4,
               group_size=8,
               model_name="Qwen/Qwen3-4B-Instruct-2507"
           ),
           learning_rate=1e-5,
           max_tokens=512,
           log_path=os.path.join(working_dir, "training_logs")
       )
       results = await tinker_train_main(config)
       return results

   if __name__ == "__main__":
       import argparse
       parser = argparse.ArgumentParser()
       parser.add_argument("--dataset_dir", required=True)
       parser.add_argument("--working_dir", required=True)
       args = parser.parse_args()

       results = asyncio.run(main(args.dataset_dir, args.working_dir))
       print(f"Training complete. Results: {{results}}")
   ```

   **DO NOT deviate from this structure. DO NOT implement train_epoch(), evaluate(), or custom training loops.**
   **Your ONLY job is to fill in: MyEnv, MyGroupBuilder, MyRLDataset, MyDatasetBuilder, and load_tasks_from_dataset_dir().**
   **Everything else must match the template exactly.**

   - Implement ONLY the required components:
     * Env class (section 2 of RL Integration Guide)
     * EnvGroupBuilder class (section 3)
     * RLDataset class (section 5 - data splitting and batching)
     * RLDatasetBuilder class with @chz.chz decorator (section 5)

   - Look for an `evaluate.py` script in the dataset directory and use it in your EnvGroupBuilder.compute_group_rewards() to calculate actual rewards.

   - If a train/test split is not already provided in the dataset, your RLDataset MUST dynamically split data into training and evaluation sets to avoid overfitting.

   - The GRPO training, model inference via Tinker API, and reward computation are all handled by tinker_cookbook.rl.train.main() - you only need to provide the Config and dataset components.

   - Save all training results (metrics, checkpoint paths, logs) to {working_dir}/results/ directory in JSON format.

   - You MUST NOT simulate model responses. tinker_cookbook.rl.train.main() handles all Tinker API calls automatically.

{sandbox_instruction}

Example invocation (paths will vary at runtime):
    python train.py --dataset_dir /path/to/dataset --working_dir /path/to/working
"""


def build_meta_prompt(
    task_files: TaskFiles,
    task_model: str,
    working_dir: str,
    provider: Provider | None = None,
    reference_dir: str | None = None,
    focus: str = "harness",
    training_sandbox: str = "modal",
) -> str:
    """Build the meta-agent prompt for creating the initial target agent.

    Args:
        focus: "harness" (default) for code improvement or "weights" for RL-based tuning
        training_sandbox: "modal" (default) or "sandboxfusion" for train.py code execution

    For Anthropic and Google providers (and the default ``None``) the text is
    byte-identical to the original. For OpenAI-compatible providers a client-setup
    block is prepended instructing the meta-agent to refactor the reference agent to
    the ``openai`` SDK at the provider's base_url/api_key_env.

    ``reference_dir`` is set only for a multi-file directory reference: instead of
    embedding the seed code, the prompt points the agent at the on-disk reference so it
    reads the files with its own tools. ``None`` (default/single-file) keeps the
    historical embedded-seed text verbatim.
    """
    # Handle weights mode (RL-based tuning)
    if focus == "weights":
        return _build_weights_meta_prompt(task_files, task_model, working_dir, training_sandbox=training_sandbox)

    # Harness mode (default - code/prompt improvement)
    reference_section = _reference_section(task_files, reference_dir)
    base = f"""You are a meta-agent. Your task is to create a target agent which can execute a task. Go ahead and create a target_agent.py for the target agent, which in turn can solve the given task.

Here is the FULL TASK SPECIFICATION that your target_agent.py will need to solve:
{task_files.task_md}

Here are a couple of sample task descriptions which the target agent has to solve:
{task_files.sample_task_descriptions}

{reference_section}

Here is a sample agent execution trajectory:
{json.dumps(task_files.sample_agent_execution, indent=2)}

{HELD_OUT_GROUND_TRUTH_NOTICE}

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
    if provider is None or provider.client_kind != "openai":
        return base
    return build_target_client_setup(provider, task_model) + base


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
    task_model: str,
    provider: Provider | None = None,
    requirements_dir: str | None = None,
    focus: str = "harness",
) -> str:
    """Build the feedback agent prompt for improving the target agent or train.py.

    focus: "harness" (default) for code improvement or "weights" for RL-based tuning

    ``requirements_dir`` is set when the reference declares dependencies (a directory
    reference, or a default/file reference shipping a requirements.txt): the agent is
    told it may add/edit a requirements.txt there. ``None`` keeps the historical text.
    """
    context_md_path = os.path.join(run_dir, "context.md")

    # Handle weights mode (RL-based tuning)
    if focus == "weights":
        return f"""You are an expert AI Engineer analyzing an RL-based agent scaffold for iterative improvement.

**GENERATION CONTEXT**:
- Current generation: {current_gen}
- Previous generations: {previous_gens}
- Evolution history: {context_md_path}

**BEFORE ANALYZING - READ THE FULL HISTORY**:
1. Read {context_md_path} to understand:
   - What improvements were tried in each previous generation
   - Training and performance trends across generations
   - What worked and what didn't work
2. Review previous improvement.md files from earlier generations if helpful
3. Don't repeat failed approaches from earlier generations
4. Build upon successful RL patterns that improved performance

---

**CURRENT TRAINING SCRIPT** (Generation {current_gen}):
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
1. improvement.md - Analysis and improvement plan for the RL pipeline
2. train.py - The improved training script

Follow these steps:

**STEP 1: Analyze the training execution**:
   - Review the RL training metrics and rewards
   - Check if the Env/EnvGroupBuilder/RLDataset are working correctly
   - Identify reward shaping issues or training instability
   - Check SandboxFusion connectivity and code execution
   - Look for convergence patterns or plateaus
   - Identify what worked well and what failed

**STEP 2: Review evolution history**:
   - Read context.md to see the full evolution of training approaches
   - Understand what RL strategies were tried in previous generations
   - Build upon successful training patterns
   - Avoid repeating failed approaches

**STEP 3: Write improvement.md**:
   - MUST save to: {next_gen_dir}/improvement.md
   - Document your analysis and planned improvements to the RL pipeline
   - Focus on improving reward signals, training efficiency, or environment design
   - Suggest better Env/EnvGroupBuilder implementations if needed
   - Reference insights from previous generations if applicable

**STEP 4: Create improved train.py**:
   - MUST save to: {next_gen_dir}/train.py
   - Implement the RL improvements documented in improvement.md
   - Maintain compatibility with tinker-cookbook RL APIs
   - Apply all the planned improvements from step 3
   - Use SandboxFusionClient with os.getenv("SANDBOX_URL") for code execution
   - Do not create or modify any other files besides these two

**RULES**:
- Focus on RL pipeline robustness and training efficiency
- Ensure proper reward shaping for your task domain
- Make the training stable across diverse problem instances
- If execution failed, fix the root cause in the RL pipeline
- Consider proper train/test data splitting and validation
- Handle edge cases and error states gracefully
- Properly log training metrics and rewards for analysis
- Ensure all RL components are correctly implemented per tinker-cookbook

NOTE: If you see errors or incomplete execution logs, focus on making the RL pipeline more robust.
"""

    # Harness mode (default - code/prompt improvement)
    base = f"""You are an expert AI Engineer analyzing agent scaffolds for iterative improvement.

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
- {HELD_OUT_GROUND_TRUTH_NOTICE}

NOTE: The agent execution log may be incomplete or contain errors if the target agent crashed. If you see an "error" field, focus on making the agent more robust to prevent such failures.
"""
    if requirements_dir is not None:
        base += (
            f"\nNOTE ON DEPENDENCIES: You may also create or edit a requirements.txt in {requirements_dir} "
            "(one package per line) to declare third-party packages your target_agent.py needs; they are "
            'installed before the target agent runs. This is the one exception to the "only two files" rule above.\n'
        )
    if provider is None or provider.client_kind != "openai":
        return base
    return build_target_client_setup(provider, task_model) + base
