#!/usr/bin/env python3
"""Reference Harbor target agent — the in-container template the meta-agent imitates.

Runs inside the benchmark task's container and leaves it in the state the verifier accepts
(no dataset, no submission file). CLI contract, provided by sia/harbor_agent.py at runtime:

    python3 target_agent.py --working_dir <dir> --instruction_file <path> --log_dir <dir>

Guarantees in-container: internet, an LLM API key in env, the model id in SIA_TASK_MODEL,
and the `anthropic` SDK installed. Drives a bash tool-use loop and records every step to
<log_dir>/agent_execution.json for the SIA feedback agent.
"""

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_TURNS = 20
COMMAND_TIMEOUT = 300

BASH_TOOL = {
    "name": "run_bash",
    "description": (
        "Run a bash command inside the task's working directory and get back its "
        "stdout, stderr and exit code. Use this to explore files, edit code, and "
        "produce the outputs the task requires."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command to execute."},
        },
        "required": ["command"],
    },
}

SYSTEM_PROMPT = (
    "You are an autonomous software engineer working inside a Linux container. "
    "You are given a task instruction and a working directory. Accomplish the task "
    "by running shell commands with the run_bash tool: inspect the environment, edit "
    "or create files, and verify your work. The task is graded by an automated "
    "verifier that checks the final state of the container, so make sure the required "
    "output files exist at the exact paths requested. When you are confident the task "
    "is fully complete, reply with a short final message and DO NOT call any more tools."
)


def _clip(text: str, limit: int = 6000) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n...(truncated, {len(text)} chars)"


def run_bash(command: str, working_dir: str) -> dict:
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
            executable="/bin/bash",
        )
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        stdout, stderr, code = "", f"command timed out after {COMMAND_TIMEOUT}s", 124
    except Exception as exc:
        stdout, stderr, code = "", f"failed to run command: {exc}", 1
    return {"stdout": _clip(stdout), "stderr": _clip(stderr), "exit_code": code}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference Harbor target agent")
    parser.add_argument("--working_dir", required=True, help="Directory to operate in")
    parser.add_argument("--instruction_file", required=True, help="Path to the task instruction")
    parser.add_argument("--log_dir", required=True, help="Where to write agent_execution.json")
    parser.add_argument("--task_model", default=None, help="Solver model id (overrides SIA_TASK_MODEL)")
    args = parser.parse_args()

    working_dir = args.working_dir
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "agent_execution.json"
    model = args.task_model or os.getenv("SIA_TASK_MODEL") or DEFAULT_MODEL
    instruction = Path(args.instruction_file).read_text(encoding="utf-8")

    trajectory = {
        "model": model,
        "working_dir": working_dir,
        "started_at": datetime.now().isoformat(),
        "instruction": instruction,
        "steps": [],
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }

    def flush() -> None:
        trajectory["finished_at"] = datetime.now().isoformat()
        log_path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")

    try:
        import anthropic
    except ImportError:
        trajectory["error"] = "anthropic SDK not available in container"
        flush()
        raise SystemExit("anthropic SDK not installed")

    client = anthropic.Anthropic()
    messages = [
        {
            "role": "user",
            "content": (
                f"Task working directory: {working_dir}\n\n"
                f"Task instruction:\n{instruction}\n\n"
                "Begin by exploring the working directory, then complete the task."
            ),
        }
    ]

    for turn in range(MAX_TURNS):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[BASH_TOOL],
            messages=messages,
        )
        usage = getattr(response, "usage", None)
        if usage:
            trajectory["usage"]["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
            trajectory["usage"]["output_tokens"] += getattr(usage, "output_tokens", 0) or 0

        assistant_content = []
        tool_uses = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
                trajectory["steps"].append({"turn": turn, "type": "assistant_text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
                tool_uses.append(block)

        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason != "tool_use" or not tool_uses:
            trajectory["stop_reason"] = response.stop_reason
            break

        tool_results = []
        for tu in tool_uses:
            command = tu.input.get("command", "")
            result = run_bash(command, working_dir)
            trajectory["steps"].append({"turn": turn, "type": "tool_use", "command": command, "result": result})
            tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": json.dumps(result)})
        messages.append({"role": "user", "content": tool_results})
    else:
        trajectory["stop_reason"] = "max_turns"

    flush()
    print(
        f"[reference_harbor_agent] done | turns={len(trajectory['steps'])} "
        f"| in={trajectory['usage']['input_tokens']} out={trajectory['usage']['output_tokens']}"
    )


if __name__ == "__main__":
    main()
