# Running SIA on Harbor benchmarks

SIA can run its self-improvement loop against external benchmarks from the
[Harbor](https://www.harborframework.com/docs) registry. Each generation's `target_agent.py` is
attached to the benchmark, executed **inside the benchmark's own Docker containers**, and scored by
Harbor's **native verifiers** — there is no local dataset or `evaluate.py`.

## How it maps onto SIA

| SIA concept | Local mode | Harbor mode |
|---|---|---|
| Task | a `tasks/<name>/` directory | a downloaded Harbor benchmark (many container tasks) |
| Running a generation | `python target_agent.py --dataset_dir … --working_dir …` | one Harbor **job**: the agent runs in every task's container |
| Scoring | local `evaluate.py` → `results.json` | each task's verifier → reward → aggregated `results.json` |
| Feedback input | execution log + `results.json` | per-task trajectories + per-task rewards (same files) |

The self-improvement loop itself is unchanged: meta-agent writes gen 1, the feedback agent reads the
scores + trajectories and rewrites the agent for gen 2, and so on.

## Quick start

```bash
uv tool install harbor          # Harbor CLI (isolated; SIA calls it as a subprocess)
harbor auth login               # one-time, for the registry
pip install 'sia-agent[claude,harbor]'
export ANTHROPIC_API_KEY="..."

# Cheap smoke test: one tiny task
sia run --harbor_dataset terminal-bench-sample@2.0 --harbor_include_task log-summary-date-ranges \
    --max_gen 2 --run_id 1

# Full benchmark
sia run --harbor_dataset terminal-bench-sample@2.0 --max_gen 3 --run_id 2
```

## Flags

| Flag | Description |
|---|---|
| `--harbor_dataset NAME@VERSION` | Benchmark downloaded from the Harbor registry (enables Harbor mode) |
| `--harbor_task_dir PATH` | Use a pre-downloaded benchmark directory (a dir whose children are task folders) |
| `--harbor_include_task NAME` | Restrict to specific task name(s); repeatable. Keeps test runs cheap. |
| `--harbor_working_dir DIR` | Container working directory the agent operates in (default `/app`) |

`SIA_HARBOR_BIN` overrides the path to the `harbor` CLI if it is not on `PATH`.

## The in-container agent contract

In Harbor mode the meta/feedback prompts are **appended** (not edited) with an in-container contract.
The generated `target_agent.py` must follow:

```
python3 target_agent.py --working_dir <dir> --instruction_file <path> --log_dir <dir>
```

Inside the container it is guaranteed: internet access, an LLM API key in the environment, the model
id in `SIA_TASK_MODEL`, and the `anthropic` SDK installed. The agent runs an agentic bash loop —
explore the working dir, edit files, verify — and leaves the container in the state the task's
verifier checks. It must **not** write a submission file. The bundled reference implementation is
[`sia/tasks/_shared/reference_harbor_agent.py`](../sia/tasks/_shared/reference_harbor_agent.py); the
meta-agent models its output on it.

## Architecture

Two small modules implement the bridge, with no changes to the agent harness:

- **`sia/harbor_agent.py`** — `SIATargetAgent`, a Harbor `BaseAgent`. For each task it uploads the
  generation's `target_agent.py` into the container, runs it against the task instruction, downloads
  the trajectory back to the host, and lets the task's verifier score the container.
- **`sia/harbor_runner.py`** — drives the `harbor` CLI as a subprocess (Harbor ships its own
  interpreter), then parses the job's `result.json` into SIA's `results.json` plus per-task
  `agent_execution/execution_q{i}.json` trajectories that the feedback agent already understands.

## Output

Per generation, under `runs/run_{id}/gen_{n}/`:

- `target_agent.py` — the agent for that generation
- `results.json` — `{score, mean_reward, n_tasks, n_solved, n_errors, per_task: [...]}`
- `agent_execution/execution_q{i}.json` — one trajectory per benchmark task
- `harbor_jobs/` — the raw Harbor job output (per-trial `result.json`, verifier reward, logs)
- `harbor_run.log` — stdout/stderr from the Harbor CLI

## Notes & limits

- Benchmark images must allow internet (the default) so the agent can install the SDK and call the
  model. The adapter installs `anthropic` best-effort; on a base image without Python it `apt`-installs
  it first.
- A task that scores `0` is still a valid run — that is exactly the signal the feedback agent uses to
  improve the next generation.
- Harbor removes task containers/images after each run, so every run re-pulls images.
