# Architecture

SIA coordinates three AI agents in a loop. Each generation, the system inspects the previous attempt, rewrites the agent, and runs it again.

## The three agents

1. **Meta-Agent** — Reads the task description and generates the initial Target Agent tailored to the task.
2. **Target Agent** — Attempts to complete the task and records its actions and results.
3. **Feedback / Improvement Agent** — Reviews the Target Agent's execution logs, identifies improvements, and rewrites the Target Agent for the next generation.

## What happens during a run

**Generation 1:**
- Meta-agent reads the task and writes `target_agent.py`
- Target agent executes the task and logs to `agent_execution.json`
- Feedback agent analyzes the run and writes an improved agent for Gen 2

**Generation 2 through N:**
- The current generation's target agent executes the task
- The feedback agent analyzes and produces the next generation
- Continues until `--max_gen` is reached

**Output:**
- All artifacts saved under `runs/run_{run_id}/gen_{n}/`
- Each generation has its own `target_agent.py` and `agent_execution.json`
- Improvement notes land in `improvement.md` (gen 2 onwards)

## Directory layout

```
sia/
├── sia/
│   ├── orchestrator.py             # Main orchestration logic
│   ├── context_manager.py          # Run/context tracking
│   ├── util.py                     # Agent runner utilities
│   ├── prepare_mlebench_dataset.py # MLE-Bench dataset preparation
│   └── tasks/                      # Bundled with the wheel
│       ├── _shared/
│       │   ├── reference_target_agent.py
│       │   └── sample_agent_execution.json
│       └── {task-id}/              # gpqa, lawbench, longcot-chess, spaceship-titanic
│           ├── data/
│           │   ├── public/         # Public dataset
│           │   │   ├── task.md         # Task description
│           │   │   └── *.csv           # Data files
│           │   └── private/        # Held-out evaluation data
│           └── reference/
│               ├── SAMPLE_TASK_DESCRIPTIONS.md
│               └── reference_target_agent.py
└── runs/                           # Generated during execution
    └── run_{id}/
        ├── venv/                   # Isolated Python environment per run
        └── gen_{n}/                # Each generation's artifacts
            ├── target_agent.py
            ├── agent_execution.json
            └── improvement.md      # gen 2 onwards
```

## Customizing prompts

The two prompts that drive self-improvement live in [`sia/orchestrator.py`](../sia/orchestrator.py):

- `META_AGENT_PROMPT` — controls how the initial Target Agent is created
- `FEEDBACK_AGENT_PROMPT` — controls how improvements are suggested
