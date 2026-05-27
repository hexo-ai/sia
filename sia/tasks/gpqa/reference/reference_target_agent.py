#!/usr/bin/env python3
"""
Gemini 3.1 Flash Lite on diamond_questions.json → generates submission JSON with model answers.

This script:
1. Loads questions from data/public/diamond_questions.json (pre-shuffled, no answers)
2. Calls Gemini API to get model predictions (letters A-D)
3. Saves answers to: results/{model}_{dataset}_{timestamp}.json

Output format:
{
  "model": "model_name",
  "dataset_config": "diamond_qna",
  "total_questions": 198,
  "timestamp": "ISO timestamp",
  "total_input_tokens": 12345,
  "total_output_tokens": 5678,
  "total_cost_usd": 1.234,
  "details": [
    {"question_id": 1, "model_answer": "A", "input_tokens": 123, "cost_usd": 0.01},
    ...
  ]
}
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from google import genai
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm as async_tqdm


# -----------------------------------------------------------------------------
# Configuration — model, labels, concurrency, pricing
# -----------------------------------------------------------------------------
MODEL_NAME = "models/gemini-3.1-flash-lite"
DATASET_LABEL = "diamond_qna"
CONCURRENCY = 1
MODEL_PRICING = {"input": 0.075, "output": 0.30}


# -----------------------------------------------------------------------------
# Structured output — schema Gemini must return as JSON (`{"answer": "A"}`)
# -----------------------------------------------------------------------------
class Answer(BaseModel):
    answer: str = Field(description="Letter A, B, C, or D")


# -----------------------------------------------------------------------------
# Cost & API client
# -----------------------------------------------------------------------------
def calculate_cost(input_tokens: int, output_tokens: int, reasoning_tokens: int = 0) -> float:
    return (input_tokens / 1e6) * MODEL_PRICING["input"] + ((output_tokens + reasoning_tokens) / 1e6) * MODEL_PRICING["output"]


def setup_gemini() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY.")
    return genai.Client(api_key=api_key)


# -----------------------------------------------------------------------------
# Prompt building & model response parsing
# -----------------------------------------------------------------------------
def format_question(example: dict) -> str:
    """
    Format a question with answer options.

    The data already has pre-shuffled options (A, B, C, D).

    Returns:
        - prompt: Formatted question prompt with options A, B, C, D
    """
    question_text = example["Question"]
    options = example["options"]

    prompt = (
        f"Answer this multiple choice question.\n\n{question_text}\n\n"
        f"A) {options['A']}\nB) {options['B']}\nC) {options['C']}\nD) {options['D']}\n\n"
        f'Respond with JSON only: {{"answer": "A"}} (value is A, B, C, or D).'
    )

    return prompt


def parse_answer_letter(model_answer_raw: str, parsed_response) -> str:
    if parsed_response is not None and hasattr(parsed_response, "answer"):
        answer = str(parsed_response.answer).strip().upper()
    else:
        try:
            answer = str(json.loads(model_answer_raw).get("answer", "")).strip().upper()
        except json.JSONDecodeError:
            answer = model_answer_raw.strip().upper()
    return answer if answer in "ABCD" else next((letter for letter in "ABCD" if letter in answer), "")


# -----------------------------------------------------------------------------
# Inference — one question (Gemini call + usage) and full run with concurrency
# -----------------------------------------------------------------------------
async def get_answer_async(
    index: int,
    example: dict,
    client: genai.Client,
    generation_config: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    Get model answer for a single question.

    Returns dict with question_id, model_answer, tokens, and cost.
    """
    question_id = example.get("id", index)
    async with semaphore:
        try:
            prompt = format_question(example)
            response, model_answer_raw, model_answer = None, "", ""
            for attempt in range(3):
                try:
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: client.models.generate_content(
                            model=MODEL_NAME, contents=prompt, config=generation_config
                        ),
                    )
                    model_answer_raw = (response.text or "").strip()
                    if not model_answer_raw:
                        raise ValueError("empty model response")
                    model_answer = parse_answer_letter(model_answer_raw, response.parsed)
                    if model_answer not in "ABCD":
                        raise ValueError(f"answer must be A–D, got: {model_answer_raw[:120]!r}")
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2**attempt)
            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", None) or 0
            output_tokens = getattr(usage, "candidates_token_count", None) or 0
            reasoning_tokens = getattr(usage, "thoughts_token_count", None) or 0
            if not output_tokens and usage and getattr(usage, "total_token_count", None):
                output_tokens = usage.total_token_count - input_tokens - reasoning_tokens
            return {
                "success": True,
                "question_id": question_id,
                "model_answer": model_answer,
                "model_answer_raw": model_answer_raw,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "cost_usd": calculate_cost(input_tokens, output_tokens, reasoning_tokens),
            }
        except Exception as exc:
            return {"success": False, "question_id": question_id, "error": str(exc)}


async def get_all_answers_async(
    questions: list, client: genai.Client, generation_config: dict, concurrency: int
) -> list:
    """Run inference on all questions concurrently."""
    semaphore = asyncio.Semaphore(max(1, concurrency))
    tasks = [
        get_answer_async(index, example, client, generation_config, semaphore)
        for index, example in enumerate(questions)
    ]
    return await async_tqdm.gather(*tasks, desc="Getting answers")


# -----------------------------------------------------------------------------
# Results — merge per-question rows into summary dict + write JSON
# -----------------------------------------------------------------------------
def build_results(questions: list, question_results: list) -> dict:
    """
    Build results JSON with model answers only (no evaluation).

    Output format matches what evaluate.py expects:
    - details: list of {question_id, model_answer, tokens, cost}
    - Summary stats: total questions, tokens, cost
    """
    results = {
        "model": MODEL_NAME,
        "dataset_config": DATASET_LABEL,
        "total_questions": len(questions),
        "errors": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_reasoning_tokens": 0,
        "total_cost_usd": 0.0,
        "details": [],
        "timestamp": datetime.now().isoformat(),
    }

    # Fields to include in each detail entry (no evaluation fields)
    detail_keys = (
        "question_id",
        "model_answer",
        "model_answer_raw",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cost_usd",
    )

    for question_result in question_results:
        if question_result.get("success"):
            results["total_input_tokens"] += question_result["input_tokens"]
            results["total_output_tokens"] += question_result["output_tokens"]
            results["total_reasoning_tokens"] += question_result["reasoning_tokens"]
            results["total_cost_usd"] += question_result["cost_usd"]
            results["details"].append({key: question_result[key] for key in detail_keys})
        else:
            results["errors"] += 1
            results["details"].append(
                {"question_id": question_result["question_id"], "error": question_result["error"]}
            )
            print(f"Error on question {question_result['question_id']}: {question_result['error']}")

    return results


def save_results_json(results: dict, output_file: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as output_handle:
        json.dump(results, output_handle, indent=2)


# -----------------------------------------------------------------------------
# Entry — load data, get answers, persist, print summary
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GPQA Reference Agent - Generate model predictions")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Path to dataset directory containing diamond_questions.json"
    )
    parser.add_argument(
        "--working_dir",
        type=Path,
        required=True,
        help="Working directory where results/ will be created"
    )
    args = parser.parse_args()

    # Construct paths from arguments
    data_file = args.dataset_dir / "diamond_questions.json"
    output_dir = args.working_dir / "results"

    if not data_file.is_file():
        raise SystemExit(f"Missing data file: {data_file}")

    # Load questions
    questions = json.loads(data_file.read_text(encoding="utf-8"))

    generation_config = {
        "temperature": 0.0,
        "max_output_tokens": 2000,
        "response_mime_type": "application/json",
        "response_schema": Answer,
    }

    client = setup_gemini()
    question_results = asyncio.run(get_all_answers_async(questions, client, generation_config, CONCURRENCY))
    results = build_results(questions, question_results)

    # Save results to working_dir/results/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{MODEL_NAME.replace('/', '_')}_{DATASET_LABEL}_{timestamp}.json"
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total_tokens = (
        results["total_input_tokens"] + results["total_output_tokens"] + results["total_reasoning_tokens"]
    )
    answered = results["total_questions"] - results["errors"]
    print(
        f"{answered}/{len(questions)} answered | "
        f"cost ${results['total_cost_usd']:.4f} | tokens {total_tokens} | saved {output_file}"
    )


if __name__ == "__main__":
    main()
