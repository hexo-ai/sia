#!/usr/bin/env python3
"""
Private evaluator for the tau2-bench task.

Unlike solution-artifact tasks, scoring a tau2-bench agent requires RE-RUNNING
the agent on held-out private episodes — we cannot just load a static solution.py.

This script:
  1. Finds target_agent.py in the generation directory
  2. Runs it against the private episodes (this directory's episodes/)
  3. Reads the predictions.json it produces
  4. Computes the pass rate and writes private_result.json

The model used to re-run the agent is read from the TASK_MODEL env var
(set by launch.sh). Falls back to the value stored in run_config.json if present.

Usage:
    python evaluate.py --gen-dir runs/run_1/gen_3
    python evaluate.py --run-dir runs/run_1
"""

import sys
import os
import json
import glob
import shutil
import tempfile
import argparse
import subprocess
import traceback
from pathlib import Path

_PRIVATE_DIR = Path(__file__).parent
_REPO_ROOT   = Path(__file__).resolve().parents[4]
_SHARED_DIR  = _REPO_ROOT / "tasks" / "_shared"


def _find_run_config(gen_dir: str) -> dict:
    """Look for run_config.json written by launch.sh in the run directory."""
    run_dir    = os.path.dirname(gen_dir)
    config_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


def _resolve_model(gen_dir: str) -> str:
    """Determine which model to use for re-running the agent."""
    # Priority: env var > run_config.json > sensible default
    model = os.environ.get("TASK_MODEL", "")
    if model:
        return model
    config = _find_run_config(gen_dir)
    model = config.get("task_model", "")
    if model:
        return model
    # Last resort fallback — works with OpenHands backend
    return "openai/gpt-4o-mini"


def run_agent_on_private(target_agent_path: str, gen_dir: str) -> tuple[str, str | None]:
    """
    Run target_agent.py against the private episode set.

    Returns (predictions_json_path, error_string_or_None).
    """
    working_dir = tempfile.mkdtemp(prefix="tau2_private_")
    model       = _resolve_model(gen_dir)
    venv_python = _find_venv_python(gen_dir)

    cmd = [
        venv_python,
        target_agent_path,
        "--dataset_dir", str(_PRIVATE_DIR),
        "--working_dir",  working_dir,
        "--shared_dir",   str(_SHARED_DIR),
        "--model",        model,
    ]

    print(f"  [private] Running agent: {' '.join(cmd[:3])} ...", flush=True)
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            timeout=1800,  # 30 min max
        )
    except subprocess.TimeoutExpired:
        return "", "Agent timed out after 30 minutes"
    except Exception as e:
        return "", f"Failed to run agent: {e}"

    predictions_path = os.path.join(working_dir, "predictions.json")
    if not os.path.exists(predictions_path):
        stderr_hint = ""
        return "", f"Agent did not produce predictions.json (return code {result.returncode}){stderr_hint}"

    return predictions_path, None


def _find_venv_python(gen_dir: str) -> str:
    """Find the run's venv python, fall back to sys.executable."""
    run_dir = os.path.dirname(gen_dir)
    candidates = [
        os.path.join(run_dir, "venv", "bin", "python"),
        os.path.join(run_dir, "..", "venv", "bin", "python"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return sys.executable


def score_predictions(predictions_path: str) -> dict:
    if not os.path.exists(predictions_path):
        return {"error": f"File not found: {predictions_path}", "score": 0.0}
    with open(predictions_path) as f:
        data = json.load(f)
    episodes = data.get("episodes", [])
    if not episodes:
        return {"error": "predictions.json contains no episodes", "score": 0.0}

    rewards  = [float(ep.get("reward", 0.0)) for ep in episodes]
    n_total  = len(rewards)
    n_passed = sum(1 for r in rewards if r >= 1.0)
    pass_rate = sum(rewards) / n_total

    print(f"  [private] {n_passed}/{n_total} passed  (pass_rate={pass_rate:.4f})", flush=True)

    return {
        "score":           pass_rate,
        "accuracy":        pass_rate,
        "lower_is_better": False,
        "n_total":         n_total,
        "n_passed":        n_passed,
        "error":           None,
    }


def evaluate_gen(gen_dir: str) -> dict:
    gen_dir = os.path.abspath(gen_dir)
    target_agent_path = os.path.join(gen_dir, "target_agent.py")

    if not os.path.exists(target_agent_path):
        return {"error": f"target_agent.py not found in {gen_dir}", "score": 0.0}

    print(f"  [private] Re-running agent on private episodes...", flush=True)
    predictions_path, error = run_agent_on_private(target_agent_path, gen_dir)

    if error:
        return {"error": error, "score": 0.0, "lower_is_better": False}

    return score_predictions(predictions_path)


def main():
    parser = argparse.ArgumentParser(description="Private evaluator — tau2-bench task")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", help="Run directory (evaluates all gen_X/)")
    group.add_argument("--gen-dir", help="Single generation directory")
    args = parser.parse_args()

    if args.gen_dir:
        gen_dir  = os.path.abspath(args.gen_dir)
        result   = evaluate_gen(gen_dir)
        out_path = os.path.join(gen_dir, "private_result.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
        print(f"\n[private] score={result['score']:.4f}  (written to {out_path})")

    elif args.run_dir:
        run_dir  = os.path.abspath(args.run_dir)
        gen_dirs = sorted(glob.glob(os.path.join(run_dir, "gen_*")))
        if not gen_dirs:
            print(f"No gen_* directories found in {run_dir}", file=sys.stderr)
            sys.exit(1)

        all_scores = {}
        for gen_dir in gen_dirs:
            gen_name = os.path.basename(gen_dir)
            print(f"\n[{gen_name}] Evaluating on private episodes...")
            result = evaluate_gen(gen_dir)
            all_scores[gen_name] = result
            out_path = os.path.join(gen_dir, "private_result.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

        output_path = os.path.join(run_dir, "private_scores.json")
        with open(output_path, "w") as f:
            json.dump(all_scores, f, indent=2)

        print(f"\n=== PRIVATE SCORES SUMMARY ===")
        for gen, r in all_scores.items():
            if not r.get("error"):
                print(f"  {gen}: score={r['score']:.4f}  ({r.get('n_passed', '?')}/{r.get('n_total', '?')} passed)")
            else:
                print(f"  {gen}: FAILED — {r['error']}")
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
