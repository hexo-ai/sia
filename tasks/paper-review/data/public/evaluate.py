#!/usr/bin/env python3
"""
Evaluate a predict_acceptance solution against the ICLR 2024 benchmark.

Usage:
    python evaluate.py solution.py

The solution.py must define a top-level `predict_acceptance(papers, **kwargs)` function.

Outputs a JSON result and writes results.json next to solution.py.
"""

import sys
import os
import json
import time
import random
import importlib.util
import traceback
from pathlib import Path

RANDOM_BASELINE = 0.5
PERFECT = 1.0
SPLIT_SEED = 42
N_TRAIN = 100
N_TEST = 100


def load_solution(solution_path: str):
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "predict_acceptance"):
        raise AttributeError(
            f"solution.py must define predict_acceptance(), not found in {solution_path}"
        )
    return module.predict_acceptance


def _split(papers: list, seed: int = SPLIT_SEED) -> tuple[list, list]:
    rng = random.Random(seed)
    accepts = [p for p in papers if p["label"] == "accept"]
    rejects = [p for p in papers if p["label"] == "reject"]
    rng.shuffle(accepts)
    rng.shuffle(rejects)
    n_each = N_TRAIN // 2
    train = accepts[:n_each] + rejects[:n_each]
    test  = accepts[n_each : n_each + N_TEST // 2] + rejects[n_each : n_each + N_TEST // 2]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def run_evaluation(predict_fn) -> dict:
    data_path = Path(__file__).parent / "iclr2024.json"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Run: bash tasks/paper-review/download_data.sh"
        )

    papers = json.loads(data_path.read_text(encoding="utf-8"))
    train_papers, test_papers = _split(papers)

    n_accept = sum(1 for p in test_papers if p["label"] == "accept")
    n_reject = sum(1 for p in test_papers if p["label"] == "reject")
    print(
        f"Data loaded: {len(train_papers)} train, {len(test_papers)} test "
        f"({n_accept} accept, {n_reject} reject in test)",
        flush=True,
    )
    print("Running predict_acceptance...", flush=True)

    train_input = [
        {"title": p["title"], "abstract": p["abstract"], "label": p["label"]}
        for p in train_papers
    ]
    test_input = [{"title": p["title"], "abstract": p["abstract"]} for p in test_papers]
    test_labels = [p["label"] for p in test_papers]

    t0 = time.time()
    raw_predictions = predict_fn(test_input, train_papers=train_input)
    elapsed = time.time() - t0
    print(f"Finished in {elapsed:.1f}s", flush=True)

    predictions = [str(p).strip().lower() for p in raw_predictions]

    if len(predictions) != len(test_labels):
        return {
            "error": f"Expected {len(test_labels)} predictions, got {len(predictions)}",
            "score": 0.0,
            "accuracy": 0.0,
        }

    invalid = [p for p in predictions if p not in ("accept", "reject")]
    if invalid:
        return {
            "error": f"Invalid predictions (must be 'accept' or 'reject'): {invalid[:5]}",
            "score": 0.0,
            "accuracy": 0.0,
        }

    accuracy = sum(p == l for p, l in zip(predictions, test_labels)) / len(test_labels)
    score = max(0.0, (accuracy - RANDOM_BASELINE) / (PERFECT - RANDOM_BASELINE))

    return {
        "accuracy": accuracy,
        "score": score,
        "elapsed_seconds": elapsed,
        "n_train": len(train_papers),
        "n_test": len(test_papers),
        "error": None,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py solution.py", file=sys.stderr)
        sys.exit(1)

    solution_path = os.path.abspath(sys.argv[1])
    if not os.path.exists(solution_path):
        print(json.dumps({"error": f"File not found: {solution_path}", "score": 0.0}))
        sys.exit(1)

    print(f"Loading solution from: {solution_path}", flush=True)

    try:
        predict_fn = load_solution(solution_path)
    except Exception as e:
        result = {"error": f"Failed to load solution: {e}", "score": 0.0, "accuracy": None}
        print(json.dumps(result))
        sys.exit(1)

    try:
        result = run_evaluation(predict_fn)
    except Exception as e:
        result = {
            "error": f"Evaluation failed: {e}\n{traceback.format_exc()}",
            "score": 0.0,
            "accuracy": None,
        }

    result["accuracy"] = result.get("accuracy", result.get("score", 0.0))
    try:
        with open(solution_path) as _sf:
            result["solution_code"] = _sf.read()
    except Exception:
        result["solution_code"] = None

    results_path = os.path.join(os.path.dirname(solution_path), "results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to: {results_path}", flush=True)

    result_for_stdout = {k: v for k, v in result.items() if k != "solution_code"}
    print("\n=== EVALUATION RESULT ===")
    print(json.dumps(result_for_stdout, indent=2))
    print("=========================")

    if result.get("error") is None:
        print(f"\nSCORE: {result['score']:.4f}")
        print(f"Accuracy: {result['accuracy']:.4f}  (random baseline = 0.50)")
    else:
        print(f"\nFAILED: {result.get('error', 'Unknown error')}")

    sys.exit(0)


if __name__ == "__main__":
    main()
