# Troubleshooting

## "Run directory already exists"

The orchestrator refuses to overwrite an existing run. Either:

- Use a different `--run_id`, or
- Delete the existing run: `rm -rf runs/run_1`

## Target agent fails during execution

Check the logs in the generation directory:

```bash
cat runs/run_1/gen_1/agent_execution.json
```

Common causes:

- Dataset paths are wrong — make sure absolute paths are used in your task
- Required Python packages are missing from the per-run venv
- `ANTHROPIC_API_KEY` (or the provider key matching your `--meta_model`) is not set

## `ImportError: No module named 'anthropic'`

SIA creates a fresh venv per run. If packages are missing:

1. Check the venv creation step in the orchestrator logs
2. Install manually into that venv:
   ```bash
   runs/run_1/venv/bin/pip install anthropic
   ```

## "No GEMINI_API_KEY environment variable set"

This only affects `prepare_mlebench_dataset.py` when it tries to generate similar task descriptions. Either:

- Set the variable: `export GEMINI_API_KEY="..."`
- Or skip that step: pass `--skip-gemini`

## `PermissionError: Kaggle authentication failed!`

`mle-bench` downloads competitions via the Kaggle API and needs credentials. Two ways to provide them:

- Env vars: `export KAGGLE_USERNAME="..." KAGGLE_KEY="..."`
- File: drop the API token from Kaggle (Account → Create New Token) at `~/.kaggle/kaggle.json`

You also need to accept the competition's rules on Kaggle's website before `mlebench prepare -c <competition>` can fetch it.
