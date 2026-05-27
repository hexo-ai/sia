# Evaluation Guide

## Overview

After each generation, the orchestrator automatically runs evaluation:

1. Target agent saves output to generation directory (e.g., `gen_1/submission.csv`)
2. Orchestrator runs: `python evaluate.py --gen-dir gen_1/`
3. Your `evaluate.py` finds the submission file, evaluates it, and saves `results.json`
4. Orchestrator loads `results.json` and adds metrics to feedback prompt

## Location

Place `evaluate.py` in: `tasks/<task_name>/data/public/evaluate.py`

## What You Need to Write

Your `evaluate.py` must have an `evaluate()` function:

```python
from pathlib import Path

def evaluate(submission_path: Path) -> dict:
    """
    Load submission file, evaluate against ground truth, return metrics.

    Args:
        submission_path: Path to the submission file (e.g., gen_1/submission.csv)

    Returns:
        dict with your metrics (any structure you want)
    """
    # 1. Load the submission file (you decide the format)
    # 2. Load ground truth from data/private/
    # 3. Compare and calculate metrics
    # 4. Return dict with results

    return {
        "accuracy": 0.85,
        "n_correct": 170,
        "n_total": 200
    }
```

## Complete Example

```python
"""Evaluate predictions against ground truth."""

import pandas as pd
from pathlib import Path

# Path to ground truth (private data)
TASK_DIR = Path(__file__).parent.parent.parent  # Go up from data/public/
TRUTH_PATH = TASK_DIR / "data/private/test.csv"


def evaluate(submission_path: Path) -> dict:
    """Evaluate submission against ground truth."""

    # Load ground truth
    truth = pd.read_csv(TRUTH_PATH)

    # Load submission
    pred = pd.read_csv(submission_path)

    # Merge and calculate accuracy
    merged = truth.merge(pred, on="id", how="left")
    correct = (merged["label"] == merged["prediction"]).sum()
    total = len(merged)
    accuracy = correct / total

    return {
        "accuracy": float(accuracy),
        "n_correct": int(correct),
        "n_total": int(total)
    }


def main():
    """For manual testing."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-dir", type=Path, required=True)
    args = parser.parse_args()

    # Find submission file (YOU handle this - check for whatever filename you expect)
    submission = args.gen_dir / "submission.csv"
    if not submission.exists():
        print(f"Error: {submission} not found")
        sys.exit(1)

    # Evaluate
    print("Evaluating...")
    results = evaluate(submission)

    # IMPORTANT: Save results.json (required by orchestrator)
    results_path = args.gen_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_path}")

    # Print summary
    print(f"Accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
```

## Key Points

1. **Orchestrator passes `--gen-dir`**: Your script receives the generation directory path
2. **YOU find the submission file**: Check for whatever filename you told the agent to create (e.g., `submission.csv`)
3. **YOU save `results.json`**: Write the results dict to `gen_dir/results.json` - orchestrator expects this file
4. **Ground truth in `data/private/`**: Store ground truth here so agents can't access it
5. **Return format is flexible**: Any dict structure works - top-level scalars will be shown in context.md

## Testing

Test before running the orchestrator:

```bash
python tasks/<task_name>/data/public/evaluate.py --gen-dir runs/run_1/gen_1
```

Make sure `results.json` gets created with your metrics!
