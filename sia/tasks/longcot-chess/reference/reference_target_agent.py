#!/usr/bin/env python3
"""
Reference target agent to run Gemini inference on chess hard questions.

This script:
1. Loads questions from chess_hard.json
2. Runs inference using Gemini API
3. Extracts and saves solutions to responses.json
4. Automatically runs evaluation and reports accuracy

Usage:
    python reference_target_agent.py --dataset_dir ./data/public --working_dir ./output

Environment variables required:
    GEMINI_API_KEY: Your Google Gemini API key
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List


def setup_gemini_client(api_key: str):
    """Initialize Gemini client."""
    from google import genai
    return genai.Client(api_key=api_key)


def load_chess_hard_questions(dataset_dir: Path) -> List[Dict[str, Any]]:
    """Load chess hard questions from chess_hard.json."""
    chess_hard_path = dataset_dir / "chess_hard.json"

    if not chess_hard_path.exists():
        raise FileNotFoundError(f"Chess hard questions not found at {chess_hard_path}")

    with open(chess_hard_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if not questions:
        raise ValueError("No questions found in chess_hard.json")

    print(f"Loaded {len(questions)} questions from chess_hard.json")
    return questions


def extract_solution(response: str):
    """Extract solution from model response.

    Handles formats like:
    - solution = ["c4", "bxc4", "bxc4"]
    - solution = 218
    """
    if not response:
        return None

    import re

    # Look for solution = [...] or solution = <number>
    pattern = r'solution\s*=\s*(.+?)(?:\n|$)'
    match = re.search(pattern, response, re.IGNORECASE)

    if not match:
        return None

    solution_str = match.group(1).strip()

    # Try to parse as list
    if solution_str.startswith('['):
        try:
            # Clean up the string and parse
            solution_str = solution_str.replace("'", '"')
            parsed = json.loads(solution_str)
            # Return as list of strings
            return [str(x).strip() for x in parsed]
        except:
            # Try to extract manually
            items = re.findall(r'[\"\']?([^,\[\]\"\'\s]+)[\"\']?', solution_str)
            return [item.strip() for item in items if item.strip()]

    # Try to parse as number or string
    solution_str = solution_str.strip('"\'')
    try:
        # Try to convert to int
        return int(solution_str)
    except ValueError:
        # Return as string
        return solution_str


def call_gemini(client, model: str, prompt: str) -> Dict[str, Any]:
    """Call Gemini API and return response with metadata."""
    from google.genai import types

    try:
        start_time = time.time()

        config = types.GenerateContentConfig(
            max_output_tokens=8192,
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )

        elapsed = time.time() - start_time

        # Extract text from response
        text_parts = []
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if not getattr(part, "thought", False) and part.text:
                    text_parts.append(part.text)

        content = "".join(text_parts) if text_parts else (response.text or "")

        # Extract usage metadata
        usage = {}
        if response.usage_metadata:
            m = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(m, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(m, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(m, "total_token_count", 0) or 0,
            }

        return {
            "success": True,
            "content": content,
            "usage": usage,
            "elapsed_seconds": elapsed,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "content": None,
            "usage": {},
            "elapsed_seconds": time.time() - start_time,
            "error": {
                "type": type(e).__name__,
                "message": str(e)
            }
        }


def run_inference(
    client,
    model: str,
    questions: List[Dict[str, Any]],
    working_dir: Path
):
    """Run inference on all questions and save results."""

    # Create working directory
    working_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print(f"\nRunning inference on {len(questions)} questions...")
    print(f"Model: {model}")
    print(f"Output directory: {working_dir}\n")

    for idx, question in enumerate(questions, 1):
        question_id = question.get("question_id", f"unknown_{idx}")
        prompt = question.get("prompt", "")

        print(f"[{idx}/{len(questions)}] Processing {question_id}...", end=" ", flush=True)

        # Call Gemini
        response = call_gemini(client, model, prompt)

        # Extract solution from response
        solution = None
        if response["success"] and response["content"]:
            solution = extract_solution(response["content"])

        # Prepare result
        result = {
            "question_id": question_id,
            "prompt": prompt,
            "solution": solution,
            "model_response": response["content"],
            "success": response["success"],
            "usage": response["usage"],
            "elapsed_seconds": response["elapsed_seconds"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        if response["error"]:
            result["error"] = response["error"]

        results.append(result)

        status = "✓" if response["success"] else "✗"
        print(f"{status} ({response['elapsed_seconds']:.1f}s)")

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    # Save all results to a single responses.json file in root working directory
    responses_file = working_dir / "responses.json"
    with open(responses_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save summary
    summary_file = working_dir / "summary.json"
    summary = {
        "total_questions": len(questions),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "total_prompt_tokens": sum(r["usage"].get("prompt_tokens", 0) for r in results),
        "total_completion_tokens": sum(r["usage"].get("completion_tokens", 0) for r in results),
        "total_elapsed_seconds": sum(r["elapsed_seconds"] for r in results),
        "model": model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total questions: {summary['total_questions']}")
    print(f"  Successful: {summary['successful']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Total tokens: {summary['total_prompt_tokens'] + summary['total_completion_tokens']}")
    print(f"  Total time: {summary['total_elapsed_seconds']:.1f}s")
    print(f"\nResults saved to: {working_dir}")
    print(f"  All responses: {responses_file}")
    print(f"  Summary: {summary_file}")
    print(f"{'='*60}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run Gemini inference on chess hard questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference on all questions
  python reference_target_agent.py --dataset_dir ./data/public --working_dir ./output

  # This will:
  # 1. Run inference on all questions from chess_hard.json
  # 2. Save results to ./output/responses.json
  # 3. Display message on how to run evaluation with evaluate.py

Environment variables:
  GEMINI_API_KEY: Your Google Gemini API key (required)
"""
    )

    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Directory containing chess_hard.json"
    )

    parser.add_argument(
        "--working_dir",
        type=Path,
        required=True,
        help="Directory to save results (responses.json and summary.json)"
    )

    args = parser.parse_args()

    # Fixed model
    model = "gemini-3.1-flash-lite-preview"

    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        print("\nPlease set your Gemini API key:")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        return 1

    # Validate dataset directory
    if not args.dataset_dir.exists():
        print(f"Error: Dataset directory does not exist: {args.dataset_dir}")
        return 1

    # Create working directory
    args.working_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Reference Target Agent - Gemini Inference")
    print(f"{'='*60}")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Working directory: {args.working_dir}")
    print(f"Model: {model}")
    print(f"{'='*60}\n")

    # Setup client
    print("Initializing Gemini client...")
    client = setup_gemini_client(api_key)

    # Load questions
    questions = load_chess_hard_questions(args.dataset_dir)

    # Run inference
    summary = run_inference(
        client,
        model,
        questions,
        args.working_dir
    )

    return 0


if __name__ == "__main__":
    exit(main())
