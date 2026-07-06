#!/usr/bin/env python3
"""Evaluate Spaceship Titanic submissions against private labels.

The evaluator expects a Kaggle-style CSV submission with two columns:
``PassengerId`` and ``Transported``. When run by SIA with ``--gen-dir``, it
writes the canonical evaluator artifact to ``gen_dir/results.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

TRUE_VALUES = {"true", "t", "1", "yes", "y"}
FALSE_VALUES = {"false", "f", "0", "no", "n"}
REQUIRED_COLUMNS = {"PassengerId", "Transported"}


def default_ground_truth_path() -> Path:
    """Return the bundled private label file for Spaceship Titanic."""
    data_dir = Path(__file__).resolve().parent.parent
    return data_dir / "private" / "test.csv"


def parse_bool(value: object) -> bool | None:
    """Parse a CSV boolean value, returning None for invalid labels."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None

    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return None


def _require_columns(fieldnames: Sequence[str] | None, path: Path) -> None:
    missing = REQUIRED_COLUMNS - set(fieldnames or [])
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{path} missing required column(s): {missing_list}")


def load_ground_truth(path: Path) -> dict[str, bool]:
    """Load private PassengerId -> Transported labels."""
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    labels: dict[str, bool] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _require_columns(reader.fieldnames, path)
        for row_num, row in enumerate(reader, start=2):
            passenger_id = (row.get("PassengerId") or "").strip()
            label = parse_bool(row.get("Transported"))
            if not passenger_id:
                raise ValueError(f"{path}:{row_num}: missing PassengerId")
            if label is None:
                raise ValueError(f"{path}:{row_num}: invalid Transported label")
            labels[passenger_id] = label
    return labels


def load_submission(path: Path) -> dict[str, bool | None]:
    """Load submission PassengerId -> predicted Transported labels."""
    if not path.is_file():
        raise FileNotFoundError(f"Submission file not found: {path}")

    predictions: dict[str, bool | None] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _require_columns(reader.fieldnames, path)
        for row in reader:
            passenger_id = (row.get("PassengerId") or "").strip()
            if passenger_id:
                predictions[passenger_id] = parse_bool(row.get("Transported"))
    return predictions


def find_submission_file(gen_dir: Path) -> Path | None:
    """Find a likely CSV submission in a generation directory."""
    if not gen_dir.is_dir():
        return None

    preferred = gen_dir / "submission.csv"
    if preferred.is_file():
        return preferred

    results_dir = gen_dir / "results"
    if results_dir.is_dir():
        result_csvs = list(results_dir.glob("*.csv"))
        if result_csvs:
            return max(result_csvs, key=lambda p: p.stat().st_mtime)

    for pattern in ("submission*.csv", "results*.csv", "output*.csv"):
        matches = list(gen_dir.glob(pattern))
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)

    csv_files = list(gen_dir.glob("*.csv"))
    if csv_files:
        return max(csv_files, key=lambda p: p.stat().st_mtime)

    return None


def evaluate_submission(submission: dict[str, bool | None], labels: dict[str, bool]) -> dict:
    """Score predictions by classification accuracy over all private labels."""
    results = {
        "total_questions": len(labels),
        "correct": 0,
        "incorrect": 0,
        "missing": 0,
        "invalid": 0,
        "extra_predictions": len(set(submission) - set(labels)),
        "accuracy": 0.0,
        "accuracy_percent": 0.0,
        "details": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    for passenger_id, expected in labels.items():
        predicted = submission.get(passenger_id)
        detail = {
            "passenger_id": passenger_id,
            "predicted": predicted,
            "is_correct": False,
        }

        if passenger_id not in submission:
            results["missing"] += 1
            detail["status"] = "missing"
        elif predicted is None:
            results["invalid"] += 1
            detail["status"] = "invalid"
        elif predicted == expected:
            results["correct"] += 1
            detail["status"] = "correct"
            detail["is_correct"] = True
        else:
            results["incorrect"] += 1
            detail["status"] = "incorrect"

        results["details"].append(detail)

    if labels:
        results["accuracy"] = results["correct"] / len(labels)
        results["accuracy_percent"] = 100 * results["accuracy"]

    return results


def _detail_to_item(detail: dict) -> dict:
    """Map an internal detail row to the leak-safe items[] contract (no gold labels)."""
    status_map = {
        "correct": "CORRECT",
        "incorrect": "WRONG",
        "missing": "MISSING",
        "invalid": "INVALID",
    }
    return {
        "id": detail["passenger_id"],
        "status": status_map.get(detail["status"], detail["status"].upper()),
        "output": detail.get("predicted"),
        "detail": detail["status"],
    }


def save_results(results: dict, output_path: Path) -> None:
    """Write evaluator results as JSON (summary scalars + items[], no gold labels)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        key: results[key]
        for key in (
            "total_questions",
            "correct",
            "incorrect",
            "missing",
            "invalid",
            "extra_predictions",
            "accuracy",
            "accuracy_percent",
            "timestamp",
        )
    }
    payload = {
        "summary": summary,
        "items": [_detail_to_item(row) for row in results["details"]],
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def print_summary(results: dict) -> None:
    """Print a compact human-readable summary."""
    print("\n" + "=" * 70)
    print("Spaceship Titanic Evaluation Results")
    print("=" * 70)
    print(f"Total Passengers:   {results['total_questions']}")
    print(f"Correct:            {results['correct']}")
    print(f"Incorrect:          {results['incorrect']}")
    print(f"Missing:            {results['missing']}")
    print(f"Invalid:            {results['invalid']}")
    print(f"Extra Predictions:  {results['extra_predictions']}")
    print(f"Accuracy:           {results['accuracy_percent']:.2f}%")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Spaceship Titanic CSV submissions")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gen-dir", type=Path, help="Generation directory containing a submission CSV")
    group.add_argument("--submission", type=Path, help="Direct path to a submission CSV file")
    parser.add_argument("--output", type=Path, default=None, help="Path to save results (default: gen-dir/results.json)")
    args = parser.parse_args()

    labels_path = default_ground_truth_path()
    print(f"Loading ground truth from: {labels_path}")
    labels = load_ground_truth(labels_path)
    print(f"Loaded {len(labels)} labels")

    if args.submission:
        submission_path = args.submission
    else:
        print(f"Searching for submission file in: {args.gen_dir}")
        submission_path = find_submission_file(args.gen_dir)
        if submission_path is None:
            raise FileNotFoundError(
                f"No submission CSV found in {args.gen_dir}. Please specify --submission path directly."
            )

    print(f"Loading submission from: {submission_path}")
    submission = load_submission(submission_path)
    print("Evaluating submission...")
    results = evaluate_submission(submission, labels)

    if args.output:
        output_path = args.output
    elif args.gen_dir:
        output_path = args.gen_dir / "results.json"
    else:
        output_path = submission_path.parent / "results.json"

    print(f"Saving results to: {output_path}")
    save_results(results, output_path)
    print_summary(results)


if __name__ == "__main__":
    main()
