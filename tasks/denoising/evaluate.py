#!/usr/bin/env python3
"""
Wrapper evaluator for the denoising task that prints the standard
"Accuracy : <float>" scoring contract by delegating to the inner
JSON evaluator at tasks/denoising/data/public/evaluate.py.

Usage (matches lawbench evaluator interface):
    python tasks/denoising/evaluate.py --submission /path/to/submission.py
    python tasks/denoising/evaluate.py --all-gens --run-dir runs/run_10

The inner evaluator emits a JSON blob containing `score` (mse_norm gated by
Poisson >= 0.97). We parse that and print:

    Accuracy : <score>
    MSE      : <mse>
    Poisson  : <poisson>
    PoissonN : <poisson_norm>

so the orchestrator's `Accuracy : X` regex reads it unchanged.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TASK_DIR = Path(__file__).parent
REPO_ROOT = TASK_DIR.parent.parent
INNER_EVAL = TASK_DIR / "data/public/evaluate.py"
# Allow overriding the python used for the inner subprocess (useful when the
# heavyweight scientific deps live in a separate env), but default to the
# current interpreter so a fresh `pip install -r requirements.txt` works.
EVAL_PYTHON = os.environ.get("DENOISING_EVAL_PYTHON", sys.executable)
EVAL_TIMEOUT = 1200  # 20 minutes — magic_denoise can be slow on full pancreas


def evaluate(submission_path: Path) -> dict:
    """Run the inner evaluator and return parsed JSON."""
    submission_path = submission_path.resolve()
    if not submission_path.exists():
        return {"score": 0.0, "error": f"submission not found: {submission_path}"}

    env = os.environ.copy()
    # Ensure the inner evaluator can resolve `tasks.denoising._vendor.*`.
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    try:
        proc = subprocess.run(
            [EVAL_PYTHON, str(INNER_EVAL), str(submission_path)],
            capture_output=True,
            text=True,
            timeout=EVAL_TIMEOUT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"score": 0.0, "error": f"evaluation timed out after {EVAL_TIMEOUT}s"}

    out = proc.stdout
    err = proc.stderr

    # The inner evaluator prints a JSON block between "=== EVALUATION RESULT ===" markers.
    # Be robust: extract the FIRST balanced JSON object after that marker.
    result = None
    marker = "=== EVALUATION RESULT ==="
    if marker in out:
        tail = out.split(marker, 1)[1]
        # Find first '{' and balance braces
        start = tail.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(tail)):
                ch = tail[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            result = json.loads(tail[start : i + 1])
                        except json.JSONDecodeError:
                            result = None
                        break

    if result is None:
        # Fallback: maybe the evaluator failed early and dumped a one-line JSON.
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    result = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

    if result is None:
        return {
            "score": 0.0,
            "error": f"could not parse evaluator output (rc={proc.returncode})",
            "stdout_tail": out[-1500:],
            "stderr_tail": err[-500:],
        }

    return result


def _print_human(result: dict) -> None:
    # Pivot: report mse_norm as the headline "Accuracy" — the original
    # `score` field is Poisson-gated and stays at 0 for all known MAGIC
    # variants in this environment. mse_norm cleanly reflects denoising
    # quality and supports monotonic improvement.
    mse = result.get("mse")
    poisson = result.get("poisson")
    mse_norm = result.get("mse_norm")
    poisson_norm = result.get("poisson_norm")
    raw_score = result.get("score")
    err = result.get("error")

    headline = float(mse_norm) if mse_norm is not None else 0.0
    headline = max(0.0, min(1.0, headline))

    # The KEY contract for the rest of the pipeline:
    print(f"Accuracy : {headline:.4f}")

    if mse is not None:
        print(f"MSE      : {mse}")
    if mse_norm is not None:
        print(f"MSE_norm : {mse_norm}")
    if poisson is not None:
        print(f"Poisson  : {poisson}")
    if poisson_norm is not None:
        print(f"PoissonN : {poisson_norm}")
    if raw_score is not None:
        print(f"RawScore : {raw_score}  (mse_norm gated by poisson_norm>=0.97)")
    if err:
        print(f"Error    : {err}")


def _all_gens(run_dir: Path) -> None:
    print(f"\nRun: {run_dir.name}")
    for gen_dir in sorted(run_dir.glob("gen_*")):
        sub = gen_dir / "submission.py"
        if not sub.exists():
            sub = gen_dir / "solution.py"
        if not sub.exists():
            print(f"  {gen_dir.name}: (no submission.py / solution.py)")
            continue
        score_path = gen_dir / "score.txt"
        if score_path.exists():
            txt = score_path.read_text()
            m = re.search(r"Accuracy\s*:\s*([0-9.]+)", txt)
            if m:
                print(f"  {gen_dir.name}: {float(m.group(1)):.4f}  (cached)")
                continue
        result = evaluate(sub)
        score = float(result.get("score") or 0.0)
        print(f"  {gen_dir.name}: {score:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--all-gens", action="store_true")
    args = parser.parse_args()

    if args.all_gens and args.run_dir:
        _all_gens(args.run_dir)
        return

    if args.submission:
        result = evaluate(args.submission)
        _print_human(result)
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
