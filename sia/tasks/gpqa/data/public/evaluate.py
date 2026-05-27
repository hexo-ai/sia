#!/usr/bin/env python3
"""
Evaluate GPQA submissions from target_agent.py output.

This script:
1. Loads ground truth from data/private/diamond_questions.json (with correct answers)
2. Loads model predictions from a submission JSON file
3. Compares model answers against correct answer letters
4. Outputs accuracy metrics and per-domain breakdown

The script automatically looks for JSON files in the results/ subdirectory
if it exists within --gen-dir.

Usage:
    python evaluate.py --gen-dir path/to/generation/directory
    python evaluate.py --submission path/to/submission.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def load_ground_truth(data_path: Path) -> List[dict]:
    """Load the ground truth questions and answers."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_correct_answers(questions: List[dict]) -> Dict[int, str]:
    """
    Build a mapping of question_id -> correct_answer_letter.

    Reads the correct_answer_letter field from the private dataset.
    """
    correct_answers = {}

    for item in questions:
        question_id = item.get("id")
        if question_id is None:
            continue

        # The private dataset already has the correct_answer_letter field
        correct_ltr = item.get("correct_answer_letter")
        if correct_ltr:
            correct_answers[question_id] = correct_ltr

    return correct_answers


def load_submission(submission_path: Path) -> Dict:
    """Load a submission JSON file."""
    if not submission_path.is_file():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")

    with open(submission_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_submission_file(gen_dir: Path) -> Optional[Path]:
    """
    Find a submission JSON file in the generation directory.

    Search order:
    1. Check for results/ subdirectory and look for JSON files there
    2. Look for common patterns in gen_dir itself (results*.json, submission*.json, etc.)
    3. If only one JSON file exists, use that
    """
    if not gen_dir.is_dir():
        return None

    # First, check if there's a results/ subdirectory (created by reference agent)
    results_dir = gen_dir / "results"
    if results_dir.is_dir():
        json_files = list(results_dir.glob("*.json"))
        if json_files:
            # Return the most recently modified file from results/
            return max(json_files, key=lambda p: p.stat().st_mtime)

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


def normalize_answer(answer: str) -> str:
    """Normalize an answer to a single letter A-D."""
    answer = str(answer).strip().upper()

    # If it's already a single letter A-D, return it
    if answer in "ABCD":
        return answer

    # Try to extract the first A-D letter from the string
    for char in answer:
        if char in "ABCD":
            return char

    return ""


def evaluate_submission(submission: Dict, correct_answers: Dict[int, str], questions: List[dict]) -> Dict:
    """
    Evaluate a submission against ground truth.

    Args:
        submission: The submission JSON containing model answers
        correct_answers: Mapping of question_id -> correct_answer_letter
        questions: Original questions list for metadata

    Returns:
        Dictionary with evaluation results
    """
    results = {
        "total_questions": len(correct_answers),
        "correct": 0,
        "incorrect": 0,
        "missing": 0,
        "invalid": 0,
        "accuracy": 0.0,
        "accuracy_percent": 0.0,
        "details": [],
        "timestamp": datetime.now().isoformat(),
    }

    # Build a lookup for question metadata
    question_lookup = {q.get("id"): q for q in questions}

    # Extract answers from submission
    # Support multiple formats
    submission_answers = {}

    if "details" in submission:
        # Format: {"details": [{"question_id": 1, "model_answer": "A"}, ...]}
        for detail in submission["details"]:
            qid = detail.get("question_id")
            answer = detail.get("model_answer", "")
            if qid is not None:
                submission_answers[qid] = normalize_answer(answer)

    elif "answers" in submission:
        # Format: {"answers": {"1": "A", "2": "B", ...}}
        for qid_str, answer in submission["answers"].items():
            try:
                qid = int(qid_str)
                submission_answers[qid] = normalize_answer(answer)
            except ValueError:
                continue

    else:
        # Try to extract from top-level if submission is just a dict of answers
        for key, value in submission.items():
            if key in ["model", "dataset_config", "timestamp", "total_questions",
                      "correct", "incorrect", "accuracy", "total_cost_usd",
                      "total_input_tokens", "total_output_tokens", "total_reasoning_tokens"]:
                continue
            try:
                qid = int(key)
                submission_answers[qid] = normalize_answer(value)
            except (ValueError, TypeError):
                continue

    # Evaluate each question
    for question_id, correct_letter in correct_answers.items():
        question = question_lookup.get(question_id, {})
        model_answer = submission_answers.get(question_id, "")

        detail = {
            "question_id": question_id,
            "correct_answer": correct_letter,
            "model_answer": model_answer,
            "domain": question.get("domain", "N/A"),
            "subdomain": question.get("subdomain", "N/A"),
        }

        if not model_answer:
            results["missing"] += 1
            detail["status"] = "missing"
            detail["is_correct"] = False
        elif model_answer not in "ABCD":
            results["invalid"] += 1
            detail["status"] = "invalid"
            detail["is_correct"] = False
        elif model_answer == correct_letter:
            results["correct"] += 1
            detail["status"] = "correct"
            detail["is_correct"] = True
        else:
            results["incorrect"] += 1
            detail["status"] = "incorrect"
            detail["is_correct"] = False

        results["details"].append(detail)

    # Calculate accuracy
    attempted = results["correct"] + results["incorrect"]
    if attempted > 0:
        results["accuracy"] = results["correct"] / attempted
        results["accuracy_percent"] = 100 * results["accuracy"]

    return results


def save_evaluation_results(results: Dict, output_path: Path) -> None:
    """Save evaluation results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


def print_summary(results: Dict) -> None:
    """Print a summary of evaluation results."""
    print("\n" + "=" * 70)
    print("GPQA Evaluation Results")
    print("=" * 70)
    print(f"Total Questions:    {results['total_questions']}")
    print(f"Correct:            {results['correct']}")
    print(f"Incorrect:          {results['incorrect']}")
    print(f"Missing:            {results['missing']}")
    print(f"Invalid:            {results['invalid']}")
    print(f"Accuracy:           {results['accuracy_percent']:.2f}%")
    print("=" * 70)

    # Print per-domain breakdown if available
    domain_stats = {}
    for detail in results["details"]:
        domain = detail.get("domain", "Unknown")
        if domain not in domain_stats:
            domain_stats[domain] = {"correct": 0, "total": 0}
        domain_stats[domain]["total"] += 1
        if detail.get("is_correct"):
            domain_stats[domain]["correct"] += 1

    if domain_stats:
        print("\nPer-Domain Accuracy:")
        print("-" * 70)
        for domain, stats in sorted(domain_stats.items()):
            accuracy = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {domain:30s} {stats['correct']:3d}/{stats['total']:3d} ({accuracy:5.1f}%)")
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GPQA submissions against ground truth"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--gen-dir",
        type=Path,
        help="Generation directory containing submission JSON"
    )
    group.add_argument(
        "--submission",
        type=Path,
        help="Direct path to submission JSON file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save evaluation results (default: gen-dir/evaluation_results.json)"
    )

    args = parser.parse_args()

    # Always load ground truth from data/private/diamond_questions.json
    # Script is in data/public/, so go up to gpqa/ then into data/private/
    script_dir = Path(__file__).resolve().parent.parent
    ground_truth_path = script_dir / "private" / "diamond_questions.json"

    # Load ground truth
    print(f"Loading ground truth from: {ground_truth_path}")
    questions = load_ground_truth(ground_truth_path)
    correct_answers = build_correct_answers(questions)
    print(f"Loaded {len(questions)} questions")

    # Determine submission path
    if args.submission:
        submission_path = args.submission
    else:
        print(f"Searching for submission file in: {args.gen_dir}")
        submission_path = find_submission_file(args.gen_dir)
        if submission_path is None:
            raise FileNotFoundError(
                f"No submission file found in {args.gen_dir}. "
                "Please specify --submission path directly."
            )

    print(f"Loading submission from: {submission_path}")
    submission = load_submission(submission_path)

    # Evaluate
    print("Evaluating submission...")
    results = evaluate_submission(submission, correct_answers, questions)

    # Determine output path
    if args.output:
        output_path = args.output
    elif args.gen_dir:
        output_path = args.gen_dir / "evaluation_results.json"
    else:
        output_path = submission_path.parent / "evaluation_results.json"

    # Save results
    print(f"Saving results to: {output_path}")
    save_evaluation_results(results, output_path)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
