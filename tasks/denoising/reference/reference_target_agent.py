#!/usr/bin/env python3
"""
Reference target agent for the scRNA-seq denoising task.

Implements an iterative agent that writes and evaluates magic_denoise functions,
using Claude to improve the solution across multiple iterations.

Usage:
    python reference_target_agent.py --dataset_dir /path/to/dataset --working_dir /path/to/working
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"

import sys as _sys
# Use the current interpreter; the denoising evaluator deps live in the repo's
# venv now that openproblems/scprep/etc. are declared in requirements.txt.
DISCOVER_PYTHON = os.environ.get("DENOISING_EVAL_PYTHON", _sys.executable)

TOOLS = [
    {
        "name": "write_file",
        "description": "Write (overwrite) a file with the given content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "read_file",
        "description": "Read and return the contents of a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "evaluate_solution",
        "description": "Evaluate the current solution.py against the benchmark. Returns JSON with score, mse, poisson metrics.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


def write_file(path: str, content: str) -> str:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} characters to '{path}'."
    except Exception as e:
        return f"Error writing file: {e}"


def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File '{path}' not found."
    except Exception as e:
        return f"Error reading file: {e}"


def evaluate_solution(working_dir: str, dataset_dir: str) -> str:
    solution_path = os.path.join(working_dir, "solution.py")
    evaluate_script = os.path.join(dataset_dir, "evaluate.py")

    if not os.path.exists(solution_path):
        return json.dumps({"error": "solution.py not found in working directory", "score": 0.0})

    try:
        result = subprocess.run(
            [DISCOVER_PYTHON, evaluate_script, solution_path],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=working_dir,
        )
        output = result.stdout
        if result.returncode != 0 and result.stderr:
            output += f"\n[stderr]\n{result.stderr}"

        # Try to extract the JSON result from the output
        lines = output.split("\n")
        json_lines = []
        in_json = False
        for line in lines:
            if line.strip().startswith("{"):
                in_json = True
            if in_json:
                json_lines.append(line)
                if line.strip() == "}":
                    break

        if json_lines:
            try:
                return "\n".join(json_lines)
            except Exception:
                pass

        return output[-3000:] if len(output) > 3000 else output

    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Evaluation timed out after 600s", "score": 0.0})
    except Exception as e:
        return json.dumps({"error": f"Evaluation failed: {e}", "score": 0.0})


def dispatch_tool(name: str, inputs: dict, working_dir: str, dataset_dir: str) -> str:
    if name == "write_file":
        return write_file(**inputs)
    elif name == "read_file":
        return read_file(**inputs)
    elif name == "evaluate_solution":
        return evaluate_solution(working_dir, dataset_dir)
    else:
        return f"Unknown tool: {name}"


def run_agent(task: str, working_dir: str, dataset_dir: str) -> list:
    print(f"\n{'='*60}")
    print(f"Starting denoising agent")
    print('='*60)

    messages = [{"role": "user", "content": task}]
    trajectory = list(messages)

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            tools=TOOLS,
            messages=messages,
        )

        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\nAssistant: {block.text[:500]}")

        if response.stop_reason == "end_turn":
            trajectory.append({"role": "assistant", "content": [
                {"type": "text", "text": block.text}
                for block in response.content if block.type == "text"
            ]})
            break

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"\n[Tool] {block.name}({list(block.input.keys())})")
                    result = dispatch_tool(block.name, block.input, working_dir, dataset_dir)
                    preview = result[:300] + "..." if len(result) > 300 else result
                    print(f"[Result] {preview}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})

            # Track trajectory
            trajectory.append({"role": "assistant", "content": [
                {
                    "type": "tool_use",
                    "name": b.name,
                    "input": b.input,
                } for b in response.content if b.type == "tool_use"
            ]})
            trajectory.append({"role": "user", "content": tool_results})
        else:
            print(f"Unexpected stop_reason: {response.stop_reason}")
            break

    print(f"\n{'='*60}\nAgent finished.\n")
    return trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="Path to the dataset/task directory (READ-ONLY)")
    parser.add_argument("--working_dir", required=True, help="Path to the working directory (WRITE here)")
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    working_dir = os.path.abspath(args.working_dir)
    os.makedirs(working_dir, exist_ok=True)

    task_md = Path(dataset_dir, "task.md").read_text(encoding="utf-8")
    solution_path = os.path.join(working_dir, "solution.py")

    task_prompt = f"""{task_md}

---

## Your Instructions

You are working on the scRNA-seq denoising task. Your goal is to write and iteratively improve a `magic_denoise` function.

**Working directory** (write all files here): `{working_dir}`
**Dataset directory** (read-only, contains task.md and evaluate.py): `{dataset_dir}`

**Workflow:**
1. Write your `magic_denoise` function to `{solution_path}` using the `write_file` tool
2. Evaluate it using the `evaluate_solution` tool (runs `evaluate.py solution.py`)
3. Read the scores and improve your implementation
4. Repeat steps 1-3 to maximize the score

**CRITICAL**: solution.py must:
- Define a top-level `magic_denoise(X, **kwargs)` function
- Return a numpy array of the same shape as input X
- All helper functions must be top-level (no closures/lambdas)
- Include all necessary imports at the top

**Available libraries**: numpy, scipy, sklearn, graphtools, scprep, scanpy, anndata, molecular_cross_validation

**Scoring**:
- Score is `mse_norm` only when `poisson_norm >= 0.97`
- Higher score = better (max 1.0)
- MAGIC baseline achieves mse_norm ≈ 0.0 (your goal is to exceed this)

Start by writing an initial `magic_denoise` implementation, then evaluate and improve it iteratively.
"""

    trajectory = run_agent(task_prompt, working_dir, dataset_dir)

    # Save execution log
    execution_log_path = os.path.join(working_dir, "agent_execution.json")
    with open(execution_log_path, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, indent=2, default=str)

    print(f"\nExecution log saved to: {execution_log_path}")


if __name__ == "__main__":
    main()
