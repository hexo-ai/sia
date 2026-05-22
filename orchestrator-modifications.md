# Orchestrator Modifications — branch `v2-sam`

Changes made to `orchestration/orchestrator.py` and `orchestration/util.py` relative to `master`, with rationale.

---

## 1. API key validation at startup

**Where:** immediately after CLI argument parsing, before any other initialization.

**What changed:**
```python
def _required_api_key(model: str) -> tuple[str, list[str]]:
    # gemini/ → GEMINI_API_KEY or GOOGLE_API_KEY
    # gpt-oss / tinker → TINKER_API_KEY
    # claude / anthropic/ → ANTHROPIC_API_KEY
    # everything else → OPENAI_API_KEY

def _check_api_key(model: str) -> None:
    # exit(1) immediately if none of the candidate keys are set

_check_api_key(meta_model)
_check_api_key(task_model)
```

**Why:** before this change, the orchestrator would start, create directories, install dependencies, and then fail silently much later when the first API call was made. Now it fails within 1 second with a clear error message before doing anything.

---

## 2. `SAMPLE_TASK_DESCRIPTIONS.md` made optional

**What changed:**
```python
# before
SAMPLE_TASK_DESCRIPTIONS = open(...).read()  # crashes if missing

# after
try:
    SAMPLE_TASK_DESCRIPTIONS = open(...).read()
except FileNotFoundError:
    SAMPLE_TASK_DESCRIPTIONS = ""
    logger.warning("...not found — proceeding without sample descriptions")
```

**Why:** this file is task-specific and may not exist for a new task. Crashing the entire run over an optional file made no sense.

---

## 3. `uv` support for venv creation and package installation

**What changed:**
```python
uv_available = subprocess.run(["which", "uv"], capture_output=True).returncode == 0
if uv_available:
    subprocess.run(["uv", "venv", "--python", "3.12", venv_dir], check=True)
else:
    venv.create(venv_dir, with_pip=True)

def pip_install(args):
    if uv_available:
        subprocess.run(["uv", "pip", "install", "--python", ...] + args, check=True)
    else:
        subprocess.run([pip_executable, "install"] + args, check=True)
```

**Why:** `uv` is 10–100x faster than `pip` for package installation. It is used automatically when available in the environment, with a transparent fallback to standard `pip`.

---

## 4. Auto-installation of task-specific dependencies

**What changed:**
```python
task_requirements = os.path.join(task_dir, "requirements.txt")
if os.path.exists(task_requirements):
    pip_install(["-r", task_requirements])
```

**Why:** each task can have its own dependencies (e.g. `tasks/denoising/requirements.txt` contains `anndata`, `scanpy`, `scprep`, etc.). Previously these had to be installed manually. Runs are now self-contained.

---

## 5. `litellm` added to base packages

**What changed:** `litellm` added to the list of packages installed in every run's venv.

**Why:** generated `target_agent.py` files may use `litellm` as a universal multi-provider client. Without this, the first run would fail at import time.

---

## 6. Explicit API key instruction in the meta-agent prompt (rule 7)

**What changed:**
```
7. The target_agent.py should use only the "{task_model}" model...
   This model requires the following environment variable(s) for authentication: {_required_api_key(task_model)[1]}.
   Read that variable from os.environ in your target_agent.py — do not hardcode any API key.
```

**Why:** without this instruction, the meta-agent had to infer which key to use from the `reference_target_agent.py`. In practice it would often copy `TINKER_API_KEY` even for Anthropic or Gemini models. Injecting the correct variable name directly into the prompt removes any ambiguity.

---

## 7. Private evaluation after each generation (display only)

**What changed:**
```python
private_eval = os.path.join(task_dir, "data/private/evaluate.py")
if os.path.exists(private_eval):
    subprocess.run([python_exec, private_eval, "--gen-dir", current_gen_directory], check=False)
    # reads private_result.json and logs the score
```

**Why:** the public score (pancreas, seed=42) is visible to the agent and can be directly optimized. The private score (PBMC + Tabula Muris Lung, seed=48,291,736) measures true generalization and must **never** be used as a training signal. It is logged for experimenter observability only. `check=False` ensures a private eval failure does not abort the run.

---

## 8. Private score plot at end of run

**What changed:**
```python
from plot_utils import plot_private_scores
plot_path = plot_private_scores(RUN_DIRECTORY, task_name=os.path.basename(task_dir))
if plot_path:
    logger.info(f"Plot saved to: {plot_path}")
```

**Why:** makes it easy to visualize how the private score (generalization) evolves across generations without having to parse JSON files manually. The plot is saved to `runs/run_X/private_scores.png`.

---

## 10. Hard time limit per generation (`--agent_timeout`)

**What changed:**
```python
# new CLI argument
parser.add_argument('--agent_timeout', type=int, default=360)

# subprocess gets killed when budget expires (entire process group via os.killpg)
_run_with_timeout(command, args.agent_timeout)

# TimeoutExpired caught separately from other failures
target_agent_error_msg = f"TIMEOUT — killed after {args.agent_timeout}s"

# meta-agent prompt rule 9
"The target agent runs under two budget constraints:
 - Hard time limit: {args.agent_timeout}s — the process is killed when exceeded.
   Save solution.py after every improvement, not only at the end.
 - Turn limit: --max_turns ..."

# feedback agent prompt — GENERATION CONTEXT section
"- Hard time limit per run: {AGENT_TIMEOUT}s — the process is killed when exceeded"

# feedback agent prompt — RULES section
"Hard time limit is {AGENT_TIMEOUT}s: the agent process is killed when exceeded.
 The agent must save its best solution to disk after every improvement, not only
 at the end. It must also budget its turns so it doesn't run indefinitely — stop
 iterating and write the final solution before the timeout hits."
```

**Why:** without a timeout, a target agent that loops indefinitely (e.g. never passes the Poisson constraint) blocks the entire run forever. The OS kill is the single source of truth — no need for separate `max_turns` or `max_iterations` inside the agent. Both the meta-agent (when generating the initial target_agent.py) and the feedback agent (when improving it) are explicitly told about the timeout so they design agents that save intermediate solutions and converge before the budget expires. The feedback agent sees `TIMEOUT` (not `FAILED`) so it can diagnose slow convergence vs actual crashes and suggest faster approaches.

---

## 11. `--include_gen0`: skip meta-agent, run reference agent first

**What changed:**
```python
parser.add_argument('--include_gen0', action='store_true',
    help='Run generation 0 using reference_target_agent.py as-is, skipping the meta-agent')

# Section 2 — first directory is gen_0 instead of gen_1
gen_num = 0 if args.include_gen0 else 1

# Section 4 — copy instead of LLM call
if args.include_gen0:
    shutil.copy(reference_agent_src, gen_0/target_agent.py)
else:
    asyncio.run(run_agent(...META_AGENT_PROMPT...))

# Main loop — starts at 0 instead of 1
loop_start = 0 if args.include_gen0 else 1
for current_gen in range(loop_start, max_gen + 1):
```

**Why:** the meta-agent is a cold-start LLM call with no execution evidence. Its only job was to bootstrap gen 1 from the reference agent. With `--include_gen0`, gen 0 runs the reference agent as-is — producing a real trajectory and score — so gen 1's feedback agent has actual execution evidence to work from. This makes the improvement loop uniform: every generation after gen 0 is a feedback agent, no special-cased meta-agent. The meta-agent path is kept as the default for backwards compatibility.

---

## 12. `--shared_dir` passed to every target agent

**What changed:**
```python
# Computed once after task_dir is resolved
ABS_SHARED_DIRECTORY = os.path.abspath(os.path.join(task_dir, "../_shared"))

# Injected into the target agent command
command = f"... python target_agent.py \
    --dataset_dir {ABS_DATASET_DIRECTORY} \
    --working_dir {current_gen_directory} \
    --shared_dir  {ABS_SHARED_DIRECTORY} ..."

# Meta-agent prompt updated: --shared_dir is now a required argument
# alongside --dataset_dir and --working_dir
```

**Why:** `tasks/_shared/call_task_model.py` provides a unified LLM caller (Tinker SDK for gpt-oss, litellm for everything else). Rather than copying it into each task's `data/public/`, target agents import it directly from `_shared/` by doing `sys.path.insert(0, shared_dir)`. This keeps a single source of truth for `call_task_model.py` and avoids per-task duplication when adding new tasks.

---

## 13. `--model` passed to every target agent

**What changed:**
```python
# Orchestrator command (was missing --model entirely)
command = f"... python target_agent.py \
    --dataset_dir {ABS_DATASET_DIRECTORY} \
    --working_dir {current_gen_directory} \
    --shared_dir  {ABS_SHARED_DIRECTORY} \
    --model       {task_model} ..."

# reference_target_agent.py (was hardcoded TINKER_MODEL = "openai/gpt-oss-120b")
parser.add_argument("--model", required=True)
content = call_task_model(messages, model=args.model, checkpoint=args.checkpoint)

# Meta-agent prompt rule 2: --model listed as required argument
# Meta-agent prompt rule 7: "read args.model and pass to call_task_model() — do NOT hardcode"
```

**Why:** Previously the model name was baked into generated code at meta-agent time (rule 7 said "use `{task_model}`" — the f-string embedded the literal model name). The reference agent also hardcoded `TINKER_MODEL`. This meant changing the task model required regenerating all agents. Now the model is a runtime parameter injected by the orchestrator, consistent with `--dataset_dir` and `--working_dir`. OpenHands still receives model-specific architectural guidelines via `TASK_MODEL_GUIDELINES` so generated agents remain model-aware.

---

## 14. Full Harmony compliance: `openai_harmony` library + multi-turn tool loop

**What changed:**

`call_llm.py` renamed to `call_task_model.py`; function renamed from `call_llm` to `call_task_model`.

Tinker path rewritten to use the official `openai_harmony` PyPI library:
```python
# Before: hand-rolled regex parser + apply_chat_template
parse_harmony(content)  # fragile regex on raw text

# After: proper token-level encoding/decoding
from openai_harmony import Conversation, Message, Role, load_harmony_encoding, HarmonyEncodingName
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
input_tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
# ... Tinker SDK samples ...
parsed = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
```

`reference_target_agent.py` restructured:
```
Before:  {"role": "user", "content": task_md + prompt}   # single user message
After:   {"role": "system",    "content": identity + date + reasoning + channels}
         {"role": "developer", "content": task instructions + TypeScript tool definitions}
         {"role": "user",      "content": "Please complete the task described above."}
```

Tool definitions moved from JSON Schema to TypeScript namespace syntax (Harmony native):
```
namespace functions {
// Run a bash command and return stdout + stderr.
type bash = (_: { command: string }) => any;
...
} // namespace functions
```

Single-shot 3-attempt loop replaced by a proper multi-turn tool loop (up to 30 turns):
```python
while turn < MAX_TURNS:
    response = call_task_model(messages, ...)
    if tool_calls: execute → append results → continue
    else:          break (final answer)
```

CoT (analysis channel) handling per spec:
- Tool call turn: keep analysis in history (model needs its reasoning context)
- Final response: drop analysis (conversation done)

`openai_harmony` added to base packages installed in every run's venv.

**Why:** our original `parse_harmony()` was a fragile regex approximation of what the official library handles correctly. The prompt was also wrong — sending everything in a single `user` message instead of the proper `system`/`developer`/`user` split. The single-shot loop couldn't handle multi-step reasoning. These gaps caused missed tool calls, malformed function arguments, and poor performance on gpt-oss-120b.

---

## 9. Silence OpenHands SDK output (`util.py`)

**What changed:**
```python
def _silence_openhands():
    # 1. Set all openhands.* loggers to CRITICAL and clear their handlers
    # 2. Return a context manager that redirects fd 1 and fd 2 to /dev/null
    #    using os.dup2 (bypasses Python-level sys.stdout/stderr redirection)

# Entire OpenHands block wrapped:
with _silence_openhands():
    llm = LLM(...)
    agent = Agent(...)
    conversation = Conversation(...)
    conversation.send_message(prompt)
    result = conversation.run()
```

**Why:** the OpenHands SDK prints its full system prompt, every agent action, and every observation to the terminal, making the orchestrator output unreadable. Two layers are needed: Python logger suppression (for `[INFO]` lines like `FileEditor initialized`) and OS fd-level redirection (for `print()` calls that bypass `sys.stdout`). Both the initialization and the run are wrapped because the SDK logs during object construction too.
