#!/usr/bin/env python3
"""
Reference target agent for the ML paper acceptance prediction task.

Uses call_task_model() from shared_dir for all LLM calls.

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--working_dir",  required=True)
    parser.add_argument("--shared_dir",   required=True, help="Path to tasks/_shared/")
    parser.add_argument("--model",        required=True,
                        help="Model name or Tinker checkpoint path")
    parser.add_argument("--max_turns",     type=int, default=30,
                        help="Maximum number of LLM turns in the tool loop (default: 30)")
    parser.add_argument("--target_agent_timeout", type=int, default=600,
                        help="Wall-clock time limit in seconds (default: 600)")
    parser.add_argument("--task_model_temperature", type=float, default=0.3,
                        help="Sampling temperature for the task model (default: 0.3)")
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

Write a `predict_acceptance(papers, **kwargs)` function and save it to `{solution_path}`.
Then evaluate it by running: python {evaluate_path} {solution_path}

**Start with the TF-IDF + logistic regression baseline** — this scores ~0.30 immediately:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def predict_acceptance(papers, **kwargs):
    train_papers = kwargs.get('train_papers', [])
    if not train_papers:
        return ['accept'] * len(papers)
    train_texts  = [p['title'] + ' ' + p['abstract'] for p in train_papers]
    train_labels = [p['label'] for p in train_papers]
    test_texts   = [p['title'] + ' ' + p['abstract'] for p in papers]
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vec.fit_transform(train_texts)
    X_test  = vec.transform(test_texts)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, train_labels)
    return list(clf.predict(X_test))
```

Evaluate this first, then iterate to improve: try richer features (abstract length, sentence
complexity, topic distribution, readability metrics), stronger classifiers (SVM, gradient boosting,
ensemble), better text preprocessing (stemming, stopwords), or feature engineering on writing style.

A per-turn status message will tell you how many turns and how much time remain — save your best
solution after every improvement.

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
        elapsed = time.time() - start_time
        remaining_turns = args.max_turns - turn + 1
        remaining_time  = args.target_agent_timeout - elapsed
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
            break

    if not os.path.exists(results_path):
        logger.error(f"No results.json after {turn} turns.")
        sys.exit(1)


if __name__ == "__main__":
    main()
