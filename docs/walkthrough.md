# Walkthrough: building a custom task

A detailed, step-by-step guide for preparing your own dataset and running SIA against it. For the short version, see the [Bring your own task](../README.md#3-bring-your-own-task) section of the README.

## Step 1: Set up the task directory

Create the layout SIA expects:

```bash
mkdir -p my-tasks/gpqa/{data/public,data/private,reference}
```

### Add your dataset and task description

Place dataset files in the appropriate folders:

```bash
# Public inputs — the agent is allowed to see these
cp questions.json my-tasks/gpqa/data/public/

# Private answers / ground truths — held out from the agent
cp answers.json my-tasks/gpqa/data/private/
```

> **Note:** The LLM is **not** told about `data/private/` during evaluation. This prevents the agent from cheating and ensures fair scoring.

Write the task description in `my-tasks/gpqa/data/public/task.md`. SIA's meta-agent reads this file to understand what to build.

### Copy the reference agent template

From a clone of this repo:

```bash
cp sia/tasks/_shared/reference_target_agent.py my-tasks/gpqa/reference/
```

### (Optional) Add sample task descriptions

Create `my-tasks/gpqa/reference/SAMPLE_TASK_DESCRIPTIONS.md` with examples of similar tasks. This helps the meta-agent generalize and reduces overfitting to the exact phrasing of `task.md`.

## Step 2: Run the orchestrator

External custom task:

```bash
sia --task_dir ./my-tasks/gpqa --max_gen 5 --run_id 1
```

Bundled task (for comparison):

```bash
sia --task gpqa --max_gen 5 --run_id 1
```

With OpenHands + Gemini:

```bash
sia \
  --task_dir ./my-tasks/gpqa \
  --max_gen 5 \
  --run_id 1 \
  --backend openhands \
  --meta_model "gemini/gemini-3.1-pro-preview"
```

See [configuration.md](configuration.md) for the full list of backends and models.

## Step 3: Analyze results

```bash
# View execution logs for a generation
cat runs/run_1/gen_1/agent_execution.json

# View improvements the feedback agent proposed
cat runs/run_1/gen_2/improvement.md

# Diff successive agent versions
diff runs/run_1/gen_1/target_agent.py runs/run_1/gen_2/target_agent.py
```

## Task directory requirements

Every task directory — bundled or custom — must look like this:

```
{task-id}/
├── data/
│   ├── public/
│   │   ├── task.md                    # Task description (orchestrator reads this)
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   └── private/
│       └── ...                        # Held-out evaluation data
└── reference/
    ├── SAMPLE_TASK_DESCRIPTIONS.md    # Similar tasks (for meta-agent context)
    └── reference_target_agent.py      # Template agent structure
```

## Preparing an MLE-Bench task

The `prepare_mlebench_dataset.py` script automates the steps above for any MLE-Bench competition. First install the extras (mle-bench is not on PyPI):

```bash
pip install 'sia-agent[mlebench]'
pip install git+https://github.com/openai/mle-bench
export KAGGLE_USERNAME="..." KAGGLE_KEY="..."   # mle-bench downloads via the Kaggle API
export GEMINI_API_KEY="..."                     # optional; required only without --skip-gemini
```

Kaggle credentials come from your account's API token (Kaggle → Account → Create New Token); the downloaded `kaggle.json` can also live at `~/.kaggle/kaggle.json` instead of env vars. Accept the competition's rules on Kaggle first or `mlebench prepare` will fail to download it.

Then run:

```bash
python -m sia.prepare_mlebench_dataset -c "spaceship-titanic"
```

This will:

1. Run `mlebench prepare -c "spaceship-titanic"`
2. Copy public and private datasets from `~/.cache/mle-bench/data/prepared/`
3. Rename `description.md` → `task.md` in `data/public/`
4. Use Gemini to generate similar task descriptions (optional)
5. Create `SAMPLE_TASK_DESCRIPTIONS.md` in `reference/`
6. Copy `reference_target_agent.py` from `_shared/` into `reference/`

**Options:**

- `--skip-gemini` — Skip the Gemini API call for similar tasks
- `--tasks-dir PATH` — Custom tasks directory (default: `./tasks`)
