#!/usr/bin/env python3
"""
Evaluate LongCoT Chess submissions from target_agent.py output.

This script:
1. Loads ground truth from data/private/answers.json (with correct answers)
2. Loads model predictions from a submission JSON file (responses.json)
3. Compares model answers against correct answers
4. Outputs accuracy metrics and per-question breakdown

The script automatically looks for JSON files in the results/ subdirectory
if it exists within --gen-dir, or searches for responses.json directly.

Usage:
    python evaluate.py --gen-dir path/to/generation/directory
    python evaluate.py --submission path/to/responses.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def load_ground_truth(answers_path: Path) -> Dict[str, Any]:
    """Load ground truth answers."""
    if not answers_path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {answers_path}")

    with open(answers_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_submission(submission_path: Path) -> List[Dict[str, Any]]:
    """Load a submission JSON file (responses.json format)."""
    if not submission_path.is_file():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")

    with open(submission_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_submission_file(gen_dir: Path) -> Optional[Path]:
    """
    Find a submission JSON file in the generation directory.

    Search order:
    1. Check for results/ subdirectory and look for JSON files there
    2. Look for responses.json directly in gen_dir
    3. Look for common patterns (results*.json, submission*.json, output*.json)
    4. If only one JSON file exists, use that
    """
    if not gen_dir.is_dir():
        return None

    # First, check if there's a results/ subdirectory
    results_dir = gen_dir / "results"
    if results_dir.is_dir():
        json_files = list(results_dir.glob("*.json"))
        if json_files:
            # Return the most recently modified file from results/
            return max(json_files, key=lambda p: p.stat().st_mtime)

    # Check for responses.json directly
    responses_file = gen_dir / "responses.json"
    if responses_file.is_file():
        return responses_file

    # Try common patterns in gen_dir itself
    patterns = ["results*.json", "submission*.json", "output*.json"]
    for pattern in patterns:
        matches = list(gen_dir.glob(pattern))
        if matches:
            # Return the most recently modified file
            return max(matches, key=lambda p: p.stat().st_mtime)

    # If no pattern matches, look for any JSON file
    json_files = list(gen_dir.glob("*.json"))
    if len(json_files) == 1:
        return json_files[0]
    elif len(json_files) > 1:
        # Return the most recently modified
        return max(json_files, key=lambda p: p.stat().st_mtime)

    return None


def evaluate_answer(
    predicted: Union[List[str], int, str, None], expected: Union[List[str], str]
) -> bool:
    """Compare predicted answer with expected answer."""
    if predicted is None:
        return False

    # Both are lists (e.g., chess moves)
    if isinstance(expected, list) and isinstance(predicted, list):
        if len(predicted) != len(expected):
            return False
        # Check if any permutation matches (for unordered answers)
        # For chess moves, order matters, so direct comparison
        return predicted == expected or set(predicted) == set(expected)

    # Both are strings/numbers
    if isinstance(expected, (str, int)) and isinstance(predicted, (str, int)):
        return str(predicted).strip() == str(expected).strip()

    # Type mismatch
    return False


def evaluate_submission(
    results: List[Dict[str, Any]], correct_answers: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate a submission against ground truth.

    Args:
        results: List of results from responses.json
        correct_answers: Dictionary of correct answers

    Returns:
        Dictionary with evaluation results
    """
    evaluation_results = []
    correct = 0
    failed_inference = 0
    wrong_answer = 0

    for result in results:
        question_id = result.get("question_id")
        predicted = result.get("solution")

        # Get expected answer
        expected = correct_answers.get(question_id)

        if predicted is None:
            # Solution is None - failed inference or parse error
            status = "FAILED_INFERENCE"
            is_correct = False
            failed_inference += 1
        else:
            # Evaluate solution
            is_correct = evaluate_answer(predicted, expected)
            if is_correct:
                status = "CORRECT"
                correct += 1
            else:
                status = "WRONG"
                wrong_answer += 1

        # Store evaluation
        eval_result = {
            "question_id": question_id,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "status": status,
        }
        evaluation_results.append(eval_result)

    # Calculate metrics
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0

    return {
        "total_questions": total,
        "correct": correct,
        "wrong_answer": wrong_answer,
        "failed_inference": failed_inference,
        "accuracy": accuracy,
        "accuracy_percent": accuracy,
        "details": evaluation_results,
        "timestamp": datetime.now().isoformat(),
    }


def save_evaluation_results(results: Dict, output_path: Path) -> None:
    """Save evaluation results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Separate summary and details for cleaner output
    output_data = {
        "summary": {
            "total_questions": results["total_questions"],
            "correct": results["correct"],
            "wrong_answer": results["wrong_answer"],
            "failed_inference": results["failed_inference"],
            "accuracy": results["accuracy"],
            "timestamp": results["timestamp"],
        },
        "results": results["details"],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)


def print_summary(results: Dict) -> None:
    """Print a summary of evaluation results."""
    print("\n" + "=" * 80)
    print("LongCoT Chess Evaluation Results")
    print("=" * 80)
    print(f"Total Questions:    {results['total_questions']}")
    print(f"Correct:            {results['correct']}")
    print(f"Wrong answer:       {results['wrong_answer']}")
    print(f"Failed inference:   {results['failed_inference']}")
    print(f"Accuracy:           {results['accuracy_percent']:.2f}%")
    print("=" * 80)

    # Print detailed results
    print("\nDetailed Results:")
    print("-" * 80)
    for idx, detail in enumerate(results["details"], 1):
        symbol = "✓" if detail["correct"] else "✗"
        question_id = detail["question_id"]
        status = detail["status"]
        print(f"[{idx:2d}/{results['total_questions']}] {question_id:30s} {symbol} {status}")
        if not detail["correct"] and detail["predicted"] is not None:
            print(f"       Expected: {detail['expected']}")
            print(f"       Got:      {detail['predicted']}")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LongCoT Chess submissions against ground truth"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--gen-dir",
        type=Path,
        help="Generation directory containing submission JSON (responses.json or results/*.json)",
    )
    group.add_argument(
        "--submission", type=Path, help="Direct path to submission JSON file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save evaluation results (default: gen-dir/results.json)",
    )

    args = parser.parse_args()

    # Always load ground truth from data/private/answers.json
    # Script is in data/public/, so go up to longcot-chess/ then into data/private/
    script_dir = Path(__file__).resolve().parent.parent
    ground_truth_path = script_dir / "private" / "answers.json"

    # Load ground truth
    print(f"Loading ground truth from: {ground_truth_path}")
    correct_answers = load_ground_truth(ground_truth_path)
    print(f"Loaded {len(correct_answers)} ground truth answers")

    # Determine submission path
    if args.submission:
        submission_path = args.submission
    else:
        print(f"Searching for submission file in: {args.gen_dir}")
        submission_path = find_submission_file(args.gen_dir)
        if submission_path is None:
            raise FileNotFoundError(
                f"No submission file found in {args.gen_dir}. "
                "Please specify --submission path directly or ensure responses.json exists."
            )

    print(f"Loading submission from: {submission_path}")
    results = load_submission(submission_path)
    print(f"Loaded {len(results)} results")

    # Evaluate
    print("\nEvaluating submission...")
    evaluation = evaluate_submission(results, correct_answers)

    # Determine output path
    if args.output:
        output_path = args.output
    elif args.gen_dir:
        output_path = args.gen_dir / "results.json"
    else:
        output_path = submission_path.parent / "results.json"

    # Save results
    print(f"\nSaving results to: {output_path}")
    save_evaluation_results(evaluation, output_path)

    # Print summary
    print_summary(evaluation)


if __name__ == "__main__":
    main()
