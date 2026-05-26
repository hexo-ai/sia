#!/usr/bin/env python3
"""
Reference target agent for the TriMul GPU kernel optimization task.

Uses call_task_model() from shared_dir for all LLM calls.
Supports gpt-oss-120b via Tinker SDK + openai_harmony and any model via litellm.

Usage:
    python reference_target_agent.py \
        --dataset_dir /path/to/data/public \
        --working_dir  /path/to/working \
        --shared_dir   /path/to/tasks/_shared \
        --model        openai/gpt-oss-120b
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


TOOLS = [
    {
        "name": "bash",
        "description": "Run a bash command and return stdout + stderr.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The shell command to run"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file and return its contents.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to the file"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file (overwrites if it exists).",
        "parameters": {
            "type": "object",
            "properties": {
                "path":    {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
]

TOOLS_TS = """\
namespace functions {

// Run a bash command and return stdout + stderr.
type bash = (_: {
// The shell command to run
command: string,
}) => any;

// Read a file and return its contents.
type read_file = (_: {
// Path to the file
path: string,
}) => any;

// Write content to a file (overwrites if it exists).
type write_file = (_: {
// Path to the file
path: string,
// Content to write
content: string,
}) => any;

} // namespace functions"""


BASELINE_SOLUTION = '''\
import torch
import torch.nn.functional as F


def trimul(Z: torch.Tensor) -> torch.Tensor:
    """Baseline: direct PyTorch translation of the reference — start here and optimize."""
    N, _, C = Z.shape
    Z_norm = F.layer_norm(Z, [C])
    a = Z_norm * torch.sigmoid(Z_norm)
    b = Z_norm * torch.sigmoid(-Z_norm)
    a_perm = a.permute(2, 0, 1)
    b_perm = b.permute(2, 1, 0)
    Y = torch.bmm(a_perm, b_perm).permute(1, 2, 0)
    mask = torch.triu(torch.ones(N, N, device=Z.device, dtype=Z.dtype))
    Y = Y * mask.unsqueeze(-1)
    g = torch.sigmoid(Z_norm)
    Y_norm = F.layer_norm(Y, [C])
    return Y_norm * g
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--working_dir",  required=True)
    parser.add_argument("--shared_dir",   required=True, help="Path to tasks/_shared/")
    parser.add_argument("--model",        required=True,
                        help="Model name or Tinker checkpoint (e.g. openai/gpt-oss-120b)")
    parser.add_argument("--max_turns",             type=int,   default=30)
    parser.add_argument("--target_agent_timeout",  type=int,   default=1200)
    parser.add_argument("--task_model_temperature", type=float, default=0.3)
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    working_dir = os.path.abspath(args.working_dir)
    os.makedirs(working_dir, exist_ok=True)

    sys.path.insert(0, args.shared_dir)
    from call_task_model import call_task_model
    import tools as _tools

    MAX_OUTPUT_CHARS = 8_000

    def _bash(command: str) -> str:
        out = _tools.bash(command, working_dir=working_dir, dataset_dir=dataset_dir)
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + f"\n... [truncated — {len(out)} total chars]"
        return out

    def _read_file(path: str) -> str:
        out = _tools.read_file(path, working_dir=working_dir, dataset_dir=dataset_dir)
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + f"\n... [truncated — {len(out)} total chars]"
        return out

    def _write_file(path: str, content: str) -> str:
        return _tools.write_file(path, content, working_dir=working_dir)

    def dispatch_tool(name: str, tool_args: dict) -> str:
        if name == "bash":
            return _bash(tool_args.get("command", ""))
        if name == "read_file":
            return _read_file(tool_args.get("path", ""))
        if name == "write_file":
            return _write_file(tool_args.get("path", ""), tool_args.get("content", ""))
        return f"[ERROR] unknown tool: {name}"

    task_md       = Path(dataset_dir, "task.md").read_text(encoding="utf-8")
    solution_path = os.path.join(working_dir, "solution.py")
    evaluate_path = os.path.join(dataset_dir, "evaluate.py")
    results_path  = os.path.join(working_dir, "results.json")
    today         = date.today().isoformat()

    system_content = f"""\
You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: {today}

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'."""

    developer_content = f"""\
# Instructions

{task_md}

Write a `trimul(Z)` function and save it to `{solution_path}`.
Then evaluate it by running: python {evaluate_path} {solution_path}

**Start immediately with this baseline** — it scores 1.0 (speedup = 1.0) and is the correct starting point to improve from:

```python
{BASELINE_SOLUTION}
```

Save this to `{solution_path}` first, evaluate it to confirm it works, then iterate.

**Quick wins to try in order:**
1. `torch.compile` — wrap the function body with `torch.compile(..., mode="max-autotune")`, can give 2-3× with zero effort
2. FP16 matmul — cast `a` and `b` to `.half()` before `torch.bmm`, cast result back to float32
3. Fused Triton kernel for the gating — one kernel reads `Z_norm` once and writes both `a` and `b`
4. Integrate mask into matmul via a custom Triton kernel

After each improvement, save the new `solution.py` and re-evaluate. Keep the best version in `{solution_path}`.
A per-turn status message will tell you how many turns and how much time remain.

Constraints:
- bash: only operate on files/folders inside the working directory or the dataset directory.
- read_file: only read files inside the working directory or the dataset directory.
- write_file: only write files inside the working directory.

Working directory (read/write): {working_dir}
Dataset directory (read-only):  {dataset_dir}

# Tools

## functions

{TOOLS_TS}"""

    messages = [
        {"role": "system",    "content": system_content},
        {"role": "developer", "content": developer_content},
    ]

    trajectory = []
    start_time = time.time()
    log_dir = os.path.join(working_dir, "task_model_logs")

    for turn in range(1, args.max_turns + 1):
        elapsed          = time.time() - start_time
        remaining_turns  = args.max_turns - turn + 1
        remaining_time   = args.target_agent_timeout - elapsed
        status = (
            "Please complete the task described above."
            if turn == 1 else
            f"[Turn {turn}/{args.max_turns} — {elapsed:.0f}s elapsed, "
            f"~{remaining_turns} turns and ~{remaining_time:.0f}s remaining]"
        )
        messages.append({"role": "user", "content": status})
        logger.info(f"Turn {turn}/{args.max_turns} — {elapsed:.0f}s elapsed")

        response = call_task_model(
            messages=messages,
            model=args.model,
            tools=TOOLS,
            log_dir=log_dir,
            temperature=args.task_model_temperature,
        )

        tool_calls = response["tool_calls"]
        logger.info(f"  tool_calls: {[tc['name'] for tc in tool_calls]}")
        if response["content"]:
            logger.info(f"  content: {response['content'][:120]}")

        if tool_calls or not response["content"]:
            messages.extend(response["raw_messages"])
        else:
            messages.extend(
                m for m in response["raw_messages"] if m.get("channel") in ("final", None)
            )

        turn_record: dict = {"turn": turn, "tool_calls": [], "content": response["content"]}

        for tc in tool_calls:
            result = dispatch_tool(tc["name"], tc["args"])
            logger.info(f"  {tc['name']} → {result[:120]}")
            turn_record["tool_calls"].append({"name": tc["name"], "args": tc["args"], "result": result})
            messages.append({
                "role":    "tool_result",
                "name":    tc["name"],
                "content": result,
                "call_id": tc.get("call_id"),
            })

        trajectory.append(turn_record)

        Path(working_dir, "agent_execution.json").write_text(
            json.dumps(trajectory, indent=2, default=str), encoding="utf-8"
        )

        if not tool_calls:
            break

        if os.path.exists(results_path):
            # Check if there's still time to keep improving
            elapsed = time.time() - start_time
            if elapsed > args.target_agent_timeout * 0.9:
                logger.info("Near timeout — stopping.")
                break
            # Otherwise keep iterating to improve the kernel

    if not os.path.exists(results_path):
        logger.error(f"No results.json after {turn} turns.")
        sys.exit(1)


if __name__ == "__main__":
    main()
