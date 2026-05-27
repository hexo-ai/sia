# SIA (Self-Improving AI)

[![arXiv](https://img.shields.io/badge/arXiv-2605.27276-b31b1b.svg)](https://arxiv.org/abs/2605.27276)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/sia-agent.svg)](https://pypi.org/project/sia-agent/)

Official implementation of [**SIA: Self Improving AI with Harness & Weight Updates**](https://arxiv.org/abs/2605.27276) (Hebbar et al., 2026) — a self-improving loop where a language-model agent updates both the harness and the weights of a task-specific agent. The paper reports a 56.6% gain on LawBench, 91.9% runtime reduction on GPU kernels, and 502% improvement on single-cell RNA denoising over baseline.

Our goal is to build a self-improving AI scientist that can autonomously go ahead and improve its performance on scientific tasks.

> **Just want to try it?** Skip to [Run SIA locally](#2-run-sia-locally-with-built-in-tasks).

### Architecture

<p align="center"><img src="docs/flow.png" alt="SIA orchestration flow" width="720"></p>
<p align="center"><i>Control flow between Meta, Target, and Feedback agents over successive generations.</i></p>

SIA operates by coordinating three main types of AI agents that work together to continuously improve task performance:

### Glossary
1. **Meta-Agent**: Reads the task description and generates an initial Target Agent tailored to the task.
2. **Target Agent**: Attempts to complete the task and records its actions and results.
3. **Feedback/Improvement Agent**: Reviews the Target Agent's performance logs, identifies improvements, and updates the Target Agent accordingly.

This iterative process allows the system to autonomously refine and enhance its ability to solve scientific tasks.


### Results

<table width="100%">
  <tr>
    <td width="50%" align="center"><img src="docs/gpqa.png" alt="GPQA Results" height="220"></td>
    <td width="50%" align="center"><img src="docs/ml_agent.png" alt="ML Agent Results" height="220"></td>
  </tr>
</table>

<p align="center"><i>Performance climbs across generations of self-improvement on GPQA and an MLE-Bench task.</i></p>

---

## Run SIA locally with built-in tasks

SIA ships with four built-in tasks: `gpqa`, `lawbench`, `longcot-chess`, `spaceship-titanic`.

### Install

Pick the Agent backend that matches the LLMs you want to run.

**Claude backend** (Claude Agent SDK, Claude models only):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install 'sia-agent[claude]'
export ANTHROPIC_API_KEY="..."
```

**OpenHands backend** (multi-provider — Gemini, OpenAI, Anthropic, etc.):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install 'sia-agent[openhands]'

# Export the key(s) for the provider(s) you'll use:
export ANTHROPIC_API_KEY="..."   # for anthropic/* models
export GEMINI_API_KEY="..."      # for gemini/* models (or GOOGLE_API_KEY)
export OPENAI_API_KEY="..."      # for openai/* models
```

Full provider/model reference: [docs/configuration.md](docs/configuration.md#api-keys).

### Run

```bash
sia --task gpqa --max_gen 5 --run_id 1
```

Swap `--task` for any of the four bundled tasks.

Artifacts land in `runs/run_{run_id}/gen_{n}/`:
- `target_agent.py` — the agent for that generation
- `agent_execution.json` — execution logs
- `improvement.md` — diff rationale (gen 2+)

### Common flags

| Flag | Default | Description |
|---|---|---|
| `--task` | — | Bundled task name (mutually exclusive with `--task_dir`) |
| `--task_dir` | — | Path to an external task directory |
| `--max_gen` | 3 | Number of self-improvement generations |
| `--run_id` | 1 | Unique run identifier |
| `--backend` | `claude` | `claude` (Claude Agent SDK) or `openhands` (multi-provider) |
| `--meta_model` | `haiku` | Meta/feedback model (e.g. `haiku`, `sonnet`, `opus`, or `gemini/...`, `openai/...` with openhands) |
| `--task_model` | `claude-haiku-4-5-20251001` | Target agent model |

Full backend, model, and API-key reference: [docs/configuration.md](docs/configuration.md). Hit a snag? [docs/troubleshooting.md](docs/troubleshooting.md).

---

## Bring your own task

Prepare a task directory with the layout below and point `--task_dir` at it:

```
my-task/
├── data/
│   ├── public/
│   │   ├── task.md          # Task description — SIA reads this
│   │   └── ...              # Inputs the agent is allowed to see
│   └── private/             # Held-out eval data; never exposed to the agent
└── reference/
    ├── reference_target_agent.py     # Template; copy from sia/tasks/_shared/
    └── SAMPLE_TASK_DESCRIPTIONS.md   # Optional: example tasks for the meta-agent
```

```bash
sia --task_dir ./my-task --max_gen 5 --run_id 1
```

**Or bring an MLE-Bench competition.** SIA can bootstrap a task directory directly from any [MLE-Bench](https://github.com/openai/mle-bench) competition — it pulls the dataset via the Kaggle API, sets up the public/private split, and drops in the reference agent template:

```bash
python -m sia.prepare_mlebench_dataset -c "spaceship-titanic"
sia --task_dir ./tasks/spaceship-titanic --max_gen 5 --run_id 1
```

Full step-by-step for both paths: [docs/walkthrough.md](docs/walkthrough.md).

---

## Further reading

- [docs/architecture.md](docs/architecture.md) — directory layout, generation flow, prompt customization
- [docs/walkthrough.md](docs/walkthrough.md) — detailed custom-task walkthrough
- [docs/configuration.md](docs/configuration.md) — backends, models, API keys, CLI reference
- [docs/troubleshooting.md](docs/troubleshooting.md) — common errors and fixes

## Citation

If you use SIA in your research, please cite:

```bibtex
@article{hebbar2026sia,
  title   = {SIA: Self Improving AI with Harness \& Weight Updates},
  author  = {Hebbar, Prannay and Manawat, Yogendra and Verboomen, Samuel and Ivanova, Alesia and Palanimalai, Selvam and Bhatia, Kunal and Baskaran, Vignesh},
  journal = {arXiv preprint arXiv:2605.27276},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.27276}
}
```
