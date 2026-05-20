#!/usr/bin/env python3
"""
Reference target agent for the scRNA-seq denoising task.

Plain chat loop — no tool calls (gpt-oss-120b via Tinker does not support them).
Each turn: model emits a ```python``` block → saved as solution.py → evaluated
in-process via evaluate.run_evaluation() → score fed back as a user message.

Usage:
    python reference_target_agent.py \
        --dataset_dir /path/to/data/public \
        --working_dir  /path/to/working \
        --model        openai/gpt-oss-120b \
        --max_turns    10
"""

import argparse
import importlib.util
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"


def _make_client(model: str) -> OpenAI:
    m = model.lower()
    if "gpt-oss" in m or "tinker" in m:
        return OpenAI(api_key=os.environ["TINKER_API_KEY"], base_url=TINKER_BASE_URL)
    if any(m.startswith(p) for p in ("claude", "anthropic/")):
        return OpenAI(api_key=os.environ["ANTHROPIC_API_KEY"],
                      base_url="https://api.anthropic.com/v1")
    if any(m.startswith(p) for p in ("gemini/", "google/")):
        key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
        return OpenAI(api_key=key,
                      base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _load_evaluate(dataset_dir: str):
    """Import run_evaluation and load_solution directly from evaluate.py."""
    path = os.path.join(dataset_dir, "evaluate.py")
    spec = importlib.util.spec_from_file_location("_evaluate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_evaluation, mod.load_solution


def _extract_code(text: str) -> str | None:
    blocks = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    return blocks[-1].strip() if blocks else None


# ── Agent loop ─────────────────────────────────────────────────────────────────

def run_agent(task: str, working_dir: str, dataset_dir: str,
              client: OpenAI, model: str, max_turns: int) -> list:
    logger.info("=" * 70)
    logger.info(f"Starting denoising agent  model={model}  max_turns={max_turns}")
    logger.info("=" * 70)

    run_evaluation, load_solution = _load_evaluate(dataset_dir)
    solution_path = os.path.join(working_dir, "solution.py")

    messages = [{"role": "user", "content": task}]
    trajectory = [{"role": "user", "content": task}]
    start_time = datetime.now()
    best_score = 0.0

    for turn in range(1, max_turns + 1):
        logger.info(f"\n{'─' * 40}  Turn {turn}/{max_turns}  {'─' * 40}")

        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=8192,
        )
        text = response.choices[0].message.content or ""
        logger.info(f"[Assistant] {text[:400]}{'...' if len(text) > 400 else ''}")

        messages.append({"role": "assistant", "content": text})
        trajectory.append({"role": "assistant", "content": text, "turn": turn})

        code = _extract_code(text)
        if code is None:
            logger.info("[Agent] No code block - done.")
            break

        os.makedirs(working_dir, exist_ok=True)
        Path(solution_path).write_text(code, encoding="utf-8")
        logger.info(f"[Agent] solution.py saved ({len(code)} chars) → evaluating...")

        try:
            magic_fn = load_solution(solution_path)
            result = run_evaluation(magic_fn)
        except Exception as e:
            result = {"error": str(e), "score": 0.0}

        score = result.get("score", 0.0)
        if score > best_score:
            best_score = score
        logger.info(f"[Eval]  score={score:.4f}  mse={result.get('mse')}  poisson={result.get('poisson')}")

        eval_json = json.dumps(result, indent=2)
        feedback = (
            f"## Evaluation result (turn {turn})\n\n"
            f"```json\n{eval_json}\n```\n\n"
            "If the score can be improved, write an updated implementation as a "
            "```python``` code block. If you are satisfied, reply without a code block."
        )
        messages.append({"role": "user", "content": feedback})
        trajectory.append({"role": "user", "content": feedback, "turn": turn, "result": result})

    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"\nAgent finished  turns={turn}  best_score={best_score:.4f}  time={duration:.1f}s")
    return trajectory


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Reference target agent — denoising task")
    parser.add_argument("--dataset_dir", required=True,
                        help="Path to data/public/ (contains task.md + evaluate.py)")
    parser.add_argument("--working_dir", required=True,
                        help="Working directory — agent writes solution.py here")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--max_turns", type=int, default=10)
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    working_dir = os.path.abspath(args.working_dir)
    os.makedirs(working_dir, exist_ok=True)

    client = _make_client(args.model)
    logger.info(f"model={args.model}  dataset={dataset_dir}  working={working_dir}")

    task_md = Path(dataset_dir, "task.md").read_text(encoding="utf-8")

    task_prompt = f"""{task_md}

---

## Your task

Write a `magic_denoise` function that achieves the highest possible score.

Each time you write code, wrap it in a ```python``` block. It will be saved as
`solution.py` and evaluated automatically — you will receive the result as feedback.
Iterate until you cannot improve further, then reply without a code block.

### Requirements

- Top-level `magic_denoise(X, **kwargs)` function
- `X`: numpy array (cells × genes, raw integer counts); accepts optional `random_state` kwarg
- Returns a numpy array of the same shape with non-negative values
- All imports included in the file

### Available libraries

numpy, scipy, sklearn, graphtools, scprep, scanpy, anndata, molecular_cross_validation

### Scoring

- Hard constraint: `poisson_norm >= 0.97` (score = 0 if not met)
- Score = `mse_norm` ∈ [0, 1] when constraint is satisfied (higher is better)

Start with a working baseline, then iterate.
"""

    trajectory = run_agent(task_prompt, working_dir, dataset_dir,
                           client, args.model, args.max_turns)

    log_path = os.path.join(working_dir, "agent_execution.json")
    Path(log_path).write_text(json.dumps(trajectory, indent=2, default=str), encoding="utf-8")
    logger.info(f"Trajectory → {log_path}")


if __name__ == "__main__":
    main()
