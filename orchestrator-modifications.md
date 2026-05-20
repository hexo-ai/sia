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
