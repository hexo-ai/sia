## Target Agent Technical Specification

Every target_agent.py — whether created from scratch or improved — must conform to the
following requirements.

### Required CLI arguments

The script MUST accept all of these via argparse:

    --dataset_dir           Absolute path to the dataset directory (READ-ONLY)
    --working_dir           Absolute path to the working directory (READ-WRITE)
    --shared_dir            Absolute path to tasks/_shared/ (add to sys.path to import call_task_model)
    --model                 Model name for all LLM calls — never hardcode this
    --max_turns             Maximum total LLM turns across all call_task_model() calls (int)
    --target_agent_timeout  Wall-clock time limit in seconds — the process is killed when exceeded
    --task_model_temperature  Sampling temperature for call_task_model() calls (float)

### File access restrictions

- Dataset directory (`--dataset_dir`): READ-ONLY. Never write to it or modify its contents.
- Working directory (`--working_dir`): READ-WRITE. All output files go here.
- No other filesystem locations may be accessed.

### Execution logging

The agent must log its full execution trajectory so the feedback agent can analyze it.
Which format to use depends on the task type:

**FOR TASKS WITH MULTIPLE INDEPENDENT SAMPLES** (e.g., 198 questions, multiple test cases):
- Create a folder: `agent_execution/` inside working_dir
- One file per sample: `execution_q0.json`, `execution_q1.json`, ... (sequential, zero-indexed)
- Each file contains the complete trajectory for that ONE sample only

**FOR TASKS WITH SINGLE EXECUTION** (e.g., build a model, analyze a dataset, produce one output):
- Save to a single file: `agent_execution.json` inside working_dir
- Contains the complete trajectory

**How to determine which format**: read the task description.
- "Independent items / multiple records to process separately" → multi-trajectory folder
- "Build a model / create one solution / optimize one system" → single file

**Format requirements (both formats)**:
- Same structure as the sample agent execution trajectory
- Include all messages, tool calls, and their results
- Valid JSON — properly close all arrays and objects
- **Write after every turn** (overwrite the file each time) so the log survives a crash
  or timeout. Do NOT write only at the end of the loop.

### Budget and timeout handling

The agent runs under two hard constraints:

**Time limit** (`--target_agent_timeout`): the orchestrator sends SIGTERM when exceeded,
then SIGKILL shortly after. The agent MUST:
- Save its best output to disk after every improvement (not only at the end)
- Stop iterating with enough margin (~90s) to write the final result cleanly
- Handle SIGTERM gracefully and exit with code 0

**Turn limit** (`--max_turns`): the total number of `call_task_model()` calls across ALL
loops and phases must not exceed this value. The agent MUST:
- Track a single shared turn counter across all call sites
- Pass a sub-budget (not `args.max_turns`) to each individual `call_task_model()` call
  when calling it more than once
- Inject a per-turn status message (user role) at each turn with turns and seconds
  remaining — do NOT put static budget numbers in the developer/system message
- When the budget is exhausted, exit cleanly (exit code 0). Do NOT call `sys.exit(1)` —
  the orchestrator treats that as a crash, not a normal completion.
