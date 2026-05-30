"""
Reference Target Agent — HiveMind Backend

Uses a local LLM via HiveMind gateway (OpenAI-compatible reverse proxy)
instead of a cloud API. Reduces token costs for high-volume iteration.

Environment variables:
  HIVEMIND_API_KEY  — API key (optional, HiveMind may not require one)
  HIVEMIND_ENDPOINT — Gateway URL (default: http://192.168.183.108:8400/v1)
  HIVEMIND_MODEL    — Model name served by HiveMind (default: glm-flash)
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("HIVEMIND_MODEL", "glm-flash")
BASE_URL = os.getenv("HIVEMIND_ENDPOINT", "http://192.168.183.108:8400/v1")
API_KEY = os.getenv("HIVEMIND_API_KEY", "hivemind")


def get_client() -> OpenAI:
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ── Tool implementations ──────────────────────────────────────────────────


def write_file(path: str, content: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to {path}"


def read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def bash(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds"


def dispatch_tool(name: str, inputs: dict) -> str:
    if name == "write_file":
        return write_file(inputs["path"], inputs["content"])
    elif name == "read_file":
        return read_file(inputs["path"])
    elif name == "bash":
        return bash(inputs["command"])
    else:
        return f"Unknown tool: {name}"


# ── Trajectory logger ──────────────────────────────────────────────────────


class MultiTrajectoryLogger:
    def __init__(self, working_dir: str):
        self.folder = Path(working_dir) / "agent_execution"
        self.folder.mkdir(exist_ok=True)
        self.trajectories: list = []

    def log_trajectory(self, trajectory_id: int, messages: list):
        filepath = self.folder / f"execution_q{trajectory_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)

    def finalize(self, total_count: int):
        print(f"Logged {total_count} trajectories to {self.folder}")


# ── Agent loop ──────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read content from a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command"},
                },
                "required": ["command"],
            },
        },
    },
]


def run_agent(task: str) -> None:
    client = get_client()
    messages = [
        {
            "role": "system",
            "content": (
                f"You are an AI agent running locally via HiveMind. "
                f"Your model is {MODEL}. "
                f"You have tools to read/write files and run bash commands. "
                f"Use them to complete the task."
            ),
        },
        {"role": "user", "content": task},
    ]

    for _ in range(50):  # max iterations
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            max_tokens=4096,
        )

        choice = response.choices[0]
        messages.append(choice.message)

        if choice.finish_reason == "stop":
            break

        if choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                fn = tool_call.function
                inputs = json.loads(fn.arguments)
                result = dispatch_tool(fn.name, inputs)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result[:2000],
                    }
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--working_dir", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    # Read task
    task_md = (dataset_dir / "task.md").read_text()

    # Run agent
    run_agent(task_md)


if __name__ == "__main__":
    main()
