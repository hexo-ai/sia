#!/usr/bin/env python3
"""
Private evaluator for the paper-review task.

Evaluates solution files against NeurIPS 2023 and ICLR 2023 — the two
held-out datasets from conferences the agent never saw during development.

Usage:
    # Score all generations in a run:
    python evaluate.py --run-dir runs/run_1

    # Score a single generation directory:
    python evaluate.py --gen-dir runs/run_1/gen_3

    # Score a single solution file:
    python evaluate.py path/to/solution.py
"""

import sys
import os
import json
import time
import glob
import random
import argparse
import importlib.util
import traceback
from pathlib import Path

_PRIVATE_SEED = 91_047_253

DATASETS = {
    "iclr2023": {},
    "iclr2022": {},
}

RANDOM_BASELINE = 0.5
PERFECT = 1.0
N_TRAIN = 100
N_TEST = 100


def load_solution(solution_path: str):
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "predict_acceptance"):
        raise AttributeError("solution.py must define predict_acceptance()")
    return module.predict_acceptance


def _split(papers: list, seed: int) -> tuple[list, list]:
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


def run_evaluation_on_dataset(predict_fn, name: str) -> dict:
    data_path = Path(__file__).parent / f"{name}.json"
    if not data_path.exists():
        return {"error": f"Dataset not found at {data_path}. Run download_data.sh", "score": 0.0}

    papers = json.loads(data_path.read_text(encoding="utf-8"))
    train_papers, test_papers = _split(papers, seed=_PRIVATE_SEED)

    train_input = [
        {"title": p["title"], "abstract": p["abstract"], "label": p["label"]}
        for p in train_papers
    ]
    test_input  = [{"title": p["title"], "abstract": p["abstract"]} for p in test_papers]
    test_labels = [p["label"] for p in test_papers]

    print(f"  [{name}] {len(train_papers)} train, {len(test_papers)} test", flush=True)

    t0 = time.time()
    try:
        raw_predictions = predict_fn(test_input, train_papers=train_input)
    except Exception as e:
        return {"error": f"predict_acceptance failed on {name}: {e}", "score": 0.0}
    elapsed = time.time() - t0

    predictions = [str(p).strip().lower() for p in raw_predictions]

    if len(predictions) != len(test_labels):
        return {"error": f"Wrong number of predictions: {len(predictions)} vs {len(test_labels)}", "score": 0.0}

    invalid = [p for p in predictions if p not in ("accept", "reject")]
    if invalid:
        return {"error": f"Invalid predictions: {invalid[:5]}", "score": 0.0}

    accuracy = sum(p == l for p, l in zip(predictions, test_labels)) / len(test_labels)
    score = max(0.0, (accuracy - RANDOM_BASELINE) / (PERFECT - RANDOM_BASELINE))

    print(
        f"  [{name}] score={score:.4f}  accuracy={accuracy:.4f}  ({elapsed:.1f}s)",
        flush=True,
    )
    return {"score": score, "accuracy": accuracy, "elapsed_seconds": elapsed, "error": None}


def score_solution(solution_path: str) -> dict:
    solution_path = os.path.abspath(solution_path)
    if not os.path.exists(solution_path):
        return {"error": f"File not found: {solution_path}", "score": 0.0}

    try:
        predict_fn = load_solution(solution_path)
    except Exception as e:
        return {"error": f"Failed to load solution: {e}", "score": 0.0}

    results = {}
    for name in DATASETS:
        results[name] = run_evaluation_on_dataset(predict_fn, name)

    valid = [r["score"] for r in results.values() if r.get("error") is None]
    avg_score = sum(valid) / len(valid) if valid else 0.0

    try:
        with open(solution_path) as _sf:
            solution_code = _sf.read()
    except Exception:
        solution_code = None

    return {
        "score": avg_score,
        "accuracy": avg_score,
        "lower_is_better": False,
        "per_dataset": results,
        "error": None if valid else "All datasets failed",
        "solution_code": solution_code,
    }


def main():
    parser = argparse.ArgumentParser(description="Private evaluator — paper-review task")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", help="Run directory (evaluates all gen_X/)")
    group.add_argument("--gen-dir", help="Single generation directory")
    group.add_argument("solution", nargs="?", help="Path to a single solution.py")
    args = parser.parse_args()

    if args.gen_dir:
        gen_dir = os.path.abspath(args.gen_dir)
        result = score_solution(os.path.join(gen_dir, "solution.py"))
        out_path = os.path.join(gen_dir, "private_result.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
        print(f"\n[private] avg SCORE: {result['score']:.4f}  (written to {out_path})")

    elif args.run_dir:
        run_dir = os.path.abspath(args.run_dir)
        gen_dirs = sorted(glob.glob(os.path.join(run_dir, "gen_*")))
        if not gen_dirs:
            print(f"No gen_* directories found in {run_dir}", file=sys.stderr)
            sys.exit(1)

        all_scores = {}
        for gen_dir in gen_dirs:
            gen_name = os.path.basename(gen_dir)
            print(f"\n[{gen_name}] Evaluating...")
            result = score_solution(os.path.join(gen_dir, "solution.py"))
            all_scores[gen_name] = result
            out_path = os.path.join(gen_dir, "private_result.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

        output_path = os.path.join(run_dir, "private_scores.json")
        with open(output_path, "w") as f:
            json.dump(all_scores, f, indent=2)

        print("\n=== PRIVATE SCORES SUMMARY ===")
        for gen, r in all_scores.items():
            if not r.get("error"):
                print(f"  {gen}: score={r['score']:.4f}")
            else:
                print(f"  {gen}: FAILED — {r['error']}")
        print(f"\nResults saved to: {output_path}")

    else:
        result = score_solution(args.solution)
        print(json.dumps(result, indent=2))
        print(f"\nAvg SCORE: {result['score']:.4f}")


if __name__ == "__main__":
    main()
