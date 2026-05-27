#!/usr/bin/env python3
"""
Evaluate a tau2-bench agent by reading its predictions.json.

Unlike solution-artifact tasks (e.g. denoising), this evaluator does NOT import
or run the agent — the agent already ran and wrote predictions.json. This script
just reads that file, computes the pass rate, and writes results.json.

Usage (called by the target agent after it finishes running all episodes):
    python evaluate.py predictions.json

Outputs results.json next to predictions.json.
"""

import sys
import os
import json
import traceback
from pathlib import Path


def score_predictions(predictions_path: str) -> dict:
    predictions_path = os.path.abspath(predictions_path)
    if not os.path.exists(predictions_path):
        return {"error": f"File not found: {predictions_path}", "score": 0.0}

    with open(predictions_path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON in predictions file: {e}", "score": 0.0}

    episodes = data.get("episodes", [])
    if not episodes:
        return {"error": "predictions.json contains no episodes", "score": 0.0}

    rewards = []
    errors  = []
    for ep in episodes:
        reward = ep.get("reward", 0.0)
        if not isinstance(reward, (int, float)):
            reward = 0.0
        rewards.append(float(reward))
        if ep.get("error"):
            errors.append({"task_id": ep.get("task_id"), "error": ep["error"]})

    n_total  = len(rewards)
    n_passed = sum(1 for r in rewards if r >= 1.0)
    n_failed = n_total - n_passed
    n_errors = len(errors)

    pass_rate = sum(rewards) / n_total if n_total > 0 else 0.0

    print(f"Episodes : {n_total}")
    print(f"Passed   : {n_passed}  ({100 * n_passed / n_total:.1f}%)")
    print(f"Failed   : {n_failed}  (errors: {n_errors})")
    print(f"Pass rate: {pass_rate:.4f}")

    result = {
        "score":         pass_rate,
        "accuracy":      pass_rate,
        "lower_is_better": False,
        "n_total":       n_total,
        "n_passed":      n_passed,
        "n_failed":      n_failed,
        "n_errors":      n_errors,
        "episode_errors": errors[:10],  # cap to avoid huge results.json
        "error":         None,
    }

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py predictions.json", file=sys.stderr)
        sys.exit(1)

    predictions_path = os.path.abspath(sys.argv[1])
    print(f"Scoring predictions: {predictions_path}", flush=True)

    try:
        result = score_predictions(predictions_path)
    except Exception as e:
        result = {
            "error": f"Scoring failed: {e}\n{traceback.format_exc()}",
            "score": 0.0,
        }

    result.setdefault("accuracy", result.get("score", 0.0))
    result.setdefault("lower_is_better", False)

    results_path = os.path.join(os.path.dirname(predictions_path), "results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to: {results_path}", flush=True)

    print("\n=== EVALUATION RESULT ===")
    print(json.dumps(result, indent=2))
    print("=========================")

    if result.get("score", 0) > 0:
        print(f"\nSCORE: {result['score']:.4f}  ({result.get('n_passed', 0)}/{result.get('n_total', 0)} passed)")
    else:
        print(f"\nFAILED: {result.get('error', 'pass rate = 0')}")

    sys.exit(0)


if __name__ == "__main__":
    main()
