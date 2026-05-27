"""
Evaluate a submission against the private ground truth.

Usage:
    python tasks/lawbench/data/public/evaluate.py --gen-dir runs/run_30/gen_1
    python tasks/lawbench/data/public/evaluate.py --all-gens --run-dir runs/run_30
    python tasks/lawbench/data/public/evaluate.py --submission runs/run_30/gen_1/submission.csv
"""

import argparse
import json
import sys
import pandas as pd
from pathlib import Path

# Get task root (go up from data/public/ to task root)
TASK_DIR   = Path(__file__).parent.parent.parent
TRUTH_PATH = TASK_DIR / "data/private/test.csv"


def evaluate(submission_path: Path) -> dict:
    truth = pd.read_csv(TRUTH_PATH)
    pred  = pd.read_csv(submission_path)

    label_col = "label" if "label" in pred.columns else pred.columns[-1]
    pred = pred.rename(columns={label_col: "pred_label"})

    # align on id
    merged = truth.merge(pred[["id", "pred_label"]], on="id", how="left")
    merged["pred_label"] = merged["pred_label"].fillna("__missing__")

    correct = merged["pred_label"].values == merged["label"].values
    acc = correct.mean()

    per_class = (
        pd.DataFrame({"true": merged["label"].values, "ok": correct})
        .groupby("true")["ok"]
        .mean()
        .sort_values()
    )

    return {
        "accuracy":  float(acc),
        "n_correct": int(correct.sum()),
        "n_total":   len(correct),
        "per_class": per_class.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, help="Path to submission CSV")
    parser.add_argument("--gen-dir",   type=Path, help="Generation directory (looks for submission.csv or predictions.csv)")
    parser.add_argument("--run-dir",   type=Path, help="Run directory (for --all-gens)")
    parser.add_argument("--runs-dir",  type=Path, help="Runs root (for --all-runs)")
    parser.add_argument("--all-gens",  action="store_true")
    parser.add_argument("--all-runs",  action="store_true")
    args = parser.parse_args()

    if args.all_runs and args.runs_dir:
        for run_dir in sorted(args.runs_dir.glob("run_*")):
            _print_all_gens(run_dir)

    elif args.all_gens and args.run_dir:
        _print_all_gens(args.run_dir)

    elif args.gen_dir:
        # Find submission file in generation directory
        submission_path = None
        for fname in ["submission.csv", "predictions.csv"]:
            candidate = args.gen_dir / fname
            if candidate.exists():
                submission_path = candidate
                print(f"Found submission: {fname}")
                break

        if not submission_path:
            print(f"Error: No submission.csv or predictions.csv found in {args.gen_dir}")
            sys.exit(1)

        # Run evaluation
        print("\nEvaluating...")
        r = evaluate(submission_path)

        # Save results to results.json (required by orchestrator)
        results_path = args.gen_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(r, f, indent=2)
        print(f"Saved results to: {results_path}")

        # Print summary
        print(f"\nAccuracy : {r['accuracy']:.4f}  ({r['n_correct']}/{r['n_total']})")
        print("\nWorst 10 classes:")
        items = sorted(r["per_class"].items(), key=lambda x: x[1])
        for cls, acc in items[:10]:
            bar = "#" * int(acc * 20)
            print(f"  {cls:<42} {acc:.2f}  {bar}")
        print("\nBest 10 classes:")
        for cls, acc in items[-10:]:
            bar = "#" * int(acc * 20)
            print(f"  {cls:<42} {acc:.2f}  {bar}")

    elif args.submission:
        r = evaluate(args.submission)
        print(f"\nAccuracy : {r['accuracy']:.4f}  ({r['n_correct']}/{r['n_total']})")
        print("\nWorst 10 classes:")
        items = sorted(r["per_class"].items(), key=lambda x: x[1])
        for cls, acc in items[:10]:
            bar = "#" * int(acc * 20)
            print(f"  {cls:<42} {acc:.2f}  {bar}")
        print("\nBest 10 classes:")
        for cls, acc in items[-10:]:
            bar = "#" * int(acc * 20)
            print(f"  {cls:<42} {acc:.2f}  {bar}")
    else:
        parser.print_help()


def _print_all_gens(run_dir: Path):
    print(f"\nRun: {run_dir.name}")
    print(f"{'Gen':<8} {'Accuracy':>10} {'Correct':>12}  File")
    print("-" * 55)
    for gen_dir in sorted(run_dir.glob("gen_*")):
        for fname in ["submission.csv", "predictions.csv"]:
            sub = gen_dir / fname
            if sub.exists():
                try:
                    r = evaluate(sub)
                    print(f"{gen_dir.name:<8} {r['accuracy']:>10.4f} {r['n_correct']:>6}/{r['n_total']:<6}  {fname}")
                except Exception as e:
                    print(f"{gen_dir.name:<8}  ERROR: {e}")
                break
    print()


if __name__ == "__main__":
    main()
