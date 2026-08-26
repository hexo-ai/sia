#!/usr/bin/env python3
"""
Qwen/Qwen3-4B-Instruct-2507 on diamond_questions.json via Tinker API.

This script:
1. Loads questions from data/public/diamond_questions.json (pre-shuffled, no answers)
2. Calls Tinker API (Qwen) to get model predictions (letters A-D)
3. Saves answers to: results/{model}_{dataset}_{timestamp}.json
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import argparse
import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm as async_tqdm


# -----------------------------------------------------------------------------
# Configuration — model, labels, concurrency, pricing
# -----------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
DATASET_LABEL = "diamond_qna"
CONCURRENCY = 5
MODEL_PRICING = {"input": 0.0, "output": 0.0}
REQUEST_TIMEOUT = 60.0
MAX_RETRIES = 2


# -----------------------------------------------------------------------------
# Structured output — schema
# -----------------------------------------------------------------------------
class Answer(BaseModel):
    answer: str = Field(description="Letter A, B, C, or D")


# -----------------------------------------------------------------------------
# Cost & API client
# -----------------------------------------------------------------------------
def calculate_cost(input_tokens: int, output_tokens: int, reasoning_tokens: int = 0) -> float:
    return (input_tokens / 1e6) * MODEL_PRICING["input"] + ((output_tokens + reasoning_tokens) / 1e6) * MODEL_PRICING["output"]


def setup_client() -> AsyncOpenAI:
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise SystemExit("Set TINKER_API_KEY environment variable.")
    return AsyncOpenAI(api_key=api_key, base_url=TINKER_BASE_URL, timeout=REQUEST_TIMEOUT)


# -----------------------------------------------------------------------------
# Prompt building & model response parsing
# -----------------------------------------------------------------------------
def format_question(example: dict) -> str:
    """
    Format a question with answer options.
    """
    question_text = example["Question"]
    options = example["options"]

    prompt = (
        f"Answer this multiple choice question.\n\n{question_text}\n\n"
        f"A) {options['A']}\nB) {options['B']}\nC) {options['C']}\nD) {options['D']}\n\n"
        f'Respond with JSON only: {{"answer": "A"}} (value is A, B, C, or D).'
    )

    return prompt


def parse_answer_letter(model_answer_raw: str) -> str:
    """
    Extract A, B, C, or D from the model response.
    """
    # Try to find JSON block
    json_match = re.search(r'\{.*\}', model_answer_raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            answer = str(data.get("answer", "")).strip().upper()
            if answer in "ABCD":
                return answer
        except json.JSONDecodeError:
            pass

    # Fallback: look for the first A, B, C, or D in the raw string
    answer = model_answer_raw.strip().upper()
    if answer in "ABCD":
        return answer
    
    # Try to find "The answer is X" pattern
    match = re.search(r'ANSWER IS ([ABCD])', answer)
    if match:
        return match.group(1)
        
    match = re.search(r'ANSWER: ([ABCD])', answer)
    if match:
        return match.group(1)

    # A few more shapes the model actually produces: "FINAL ANSWER - B",
    # "**C**", "(D)", or a reply that is nothing but the letter and punctuation.
    for pattern in (r'FINAL ANSWER\s*[:\-]?\s*\(?\*{0,2}([ABCD])',
                    r'\*\*\s*([ABCD])\s*\*\*',
                    r'\(\s*([ABCD])\s*\)',
                    r'^\W*([ABCD])\W*$'):
        match = re.search(pattern, answer, re.MULTILINE)
        if match:
            return match.group(1)

    # No stated answer. Return "" rather than scanning the text for a bare
    # letter: `next(l for l in "ABCD" if l in answer)` matches letters inside
    # ordinary words, so any reply containing "A" -- including a refusal or an
    # error message -- scores as "A". That silently converts unanswered items
    # into confident wrong answers, and the harness cannot tell the difference
    # afterwards. An empty string is a measurable abstention; a guess is not.
    return ""


# -----------------------------------------------------------------------------
# Inference — one question
# -----------------------------------------------------------------------------
async def get_answer_async(
    index: int,
    example: dict,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    Get model answer for a single question.
    """
    question_id = example.get("id", index)
    async with semaphore:
        try:
            prompt = format_question(example)
            model_answer_raw, model_answer = "", ""
            input_tokens, output_tokens = 0, 0

            for attempt in range(MAX_RETRIES + 1):
                try:
                    response = await client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=1000,
                        timeout=REQUEST_TIMEOUT,
                        # Some models might not support json_object mode, but Tinker usually does
                        response_format={"type": "json_object"}
                    )
                    model_answer_raw = (response.choices[0].message.content or "").strip()
                    if not model_answer_raw:
                        raise ValueError("empty model response")
                    
                    model_answer = parse_answer_letter(model_answer_raw)
                    if model_answer not in "ABCD":
                        raise ValueError(f"answer must be A–D, got: {model_answer_raw[:120]!r}")
                    
                    usage = response.usage
                    input_tokens = usage.prompt_tokens
                    output_tokens = usage.completion_tokens
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES:
                        raise
                    await asyncio.sleep(2**attempt)
            
            return {
                "success": True,
                "question_id": question_id,
                "model_answer": model_answer,
                "model_answer_raw": model_answer_raw,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": 0,
                "cost_usd": calculate_cost(input_tokens, output_tokens),
            }
        except Exception as exc:
            return {"success": False, "question_id": question_id, "error": str(exc)}


async def get_all_answers_async(
    questions: list, client: AsyncOpenAI, concurrency: int
) -> list:
    """Run inference on all questions concurrently."""
    semaphore = asyncio.Semaphore(max(1, concurrency))
    tasks = [
        get_answer_async(index, example, client, semaphore)
        for index, example in enumerate(questions)
    ]
    return await async_tqdm.gather(*tasks, desc="Getting answers")


# -----------------------------------------------------------------------------
# Results — merge per-question rows into summary dict
# -----------------------------------------------------------------------------
def build_results(questions: list, question_results: list) -> dict:
    """
    Build results JSON with model answers only.
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


# -----------------------------------------------------------------------------
# Entry — load data, get answers, persist, print summary
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GPQA Reference Agent - Qwen via Tinker")
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

    data_file = args.dataset_dir / "diamond_questions.json"
    output_dir = args.working_dir / "results"

    if not data_file.is_file():
        raise SystemExit(f"Missing data file: {data_file}")

    # Load questions
    questions = json.loads(data_file.read_text(encoding="utf-8"))

    client = setup_client()
    question_results = asyncio.run(get_all_answers_async(questions, client, CONCURRENCY))
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
