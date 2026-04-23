import anthropic
import subprocess
import json
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"

# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "write_file",
        "description": "Write (overwrite) a file with the given content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string", "description": "File path to write"},
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
        "name": "bash",
        "description": "Run a bash command and return stdout + stderr.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
            },
            "required": ["command"],
        },
    },
]

# ── Tool implementations ──────────────────────────────────────────────────────

def write_file(path: str, content: str) -> str:
    try:
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


def bash(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error running command: {e}"


def dispatch_tool(name: str, inputs: dict) -> str:
    if name == "write_file":
        return write_file(**inputs)
    elif name == "read_file":
        return read_file(**inputs)
    elif name == "bash":
        return bash(**inputs)
    else:
        return f"Unknown tool: {name}"


# ── Multi-Trajectory Logger ───────────────────────────────────────────────────

class MultiTrajectoryLogger:
    """
    Logger for tasks with multiple independent samples (e.g., GPQA with multiple questions).

    For tasks where you need to process multiple independent items (questions, test cases,
    samples), this logger saves each trajectory separately instead of one large file.

    Usage:
        logger = MultiTrajectoryLogger(working_dir)

        for idx, question in enumerate(questions):
            messages = []
            messages.append({"role": "user", "content": question_prompt})

            response = client.messages.create(...)
            messages.append({"role": "assistant", "content": response.content})

            # Save this trajectory
            logger.log_trajectory(idx, messages)

        logger.finalize(len(questions))
    """

    def __init__(self, working_dir: str):
        """
        Initialize the multi-trajectory logger.

        Args:
            working_dir: Path to the working directory where agent_execution/ will be created
        """
        import os
        self.working_dir = working_dir
        self.execution_folder = os.path.join(working_dir, "agent_execution")
        os.makedirs(self.execution_folder, exist_ok=True)
        print(f"Initialized multi-trajectory logger at: {self.execution_folder}")

    def log_trajectory(self, trajectory_id: int, messages: list):
        """
        Save a complete trajectory for one sample.

        Args:
            trajectory_id: Index of this trajectory (0-based)
            messages: List of message dicts (same format as Anthropic API messages)
        """
        import os
        filename = f"execution_q{trajectory_id}.json"
        filepath = os.path.join(self.execution_folder, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved trajectory {trajectory_id} to {filename}")
        except Exception as e:
            print(f"  ✗ Error saving trajectory {trajectory_id}: {e}")

    def finalize(self, total_count: int):
        """
        Log completion message.

        Args:
            total_count: Total number of trajectories saved
        """
        print(f"\n{'='*60}")
        print(f"✓ Multi-trajectory logging complete:")
        print(f"  - Total trajectories: {total_count}")
        print(f"  - Saved to: {self.execution_folder}/")
        print(f"  - Files: execution_q0.json to execution_q{total_count-1}.json")
        print(f"{'='*60}\n")


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(task: str) -> None:
    print(f"\n{'='*60}")
    print(f"Task: {task}")
    print('='*60)

    messages = [{"role": "user", "content": task}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            tools=TOOLS,
            messages=messages,
        )

        # Collect any text the model emits this turn
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\nAssistant: {block.text}")

        # Done – no more tool calls
        if response.stop_reason == "end_turn":
            break

        # Handle tool calls
        if response.stop_reason == "tool_use":
            # Append assistant turn
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"\n[Tool] {block.name}({json.dumps(block.input, ensure_ascii=False)})")
                    result = dispatch_tool(block.name, block.input)
                    print(f"[Result] {result[:200]}{'...' if len(result) > 200 else ''}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})
        else:
            # Unexpected stop reason – bail out
            print(f"Unexpected stop_reason: {response.stop_reason}")
            break

    print(f"\n{'='*60}\nDone.\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_agent(
        "Write a Python file called hello.py that prints 'Hello, World!', "
        "then run it with bash and confirm the output is correct."
    )


# ── Multi-Trajectory Usage Example ────────────────────────────────────────────
"""
USAGE EXAMPLE: Multi-Trajectory Logging for GPQA-style tasks

For tasks with multiple independent questions/samples (like GPQA with 198 questions),
use MultiTrajectoryLogger instead of saving to a single agent_execution.json file.

Example implementation:

    import argparse
    import os

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--working_dir', required=True)
    args = parser.parse_args()

    # Initialize multi-trajectory logger
    logger = MultiTrajectoryLogger(args.working_dir)

    # Load dataset (e.g., GPQA questions)
    questions_file = os.path.join(args.dataset_dir, "diamond_qna.json")
    with open(questions_file) as f:
        questions = json.load(f)

    # Process each question independently
    for idx, question_data in enumerate(questions):
        print(f"\\nProcessing question {idx+1}/{len(questions)}...")

        # Build conversation for this question
        messages = []
        messages.append({
            "role": "user",
            "content": f"Question: {question_data['question']}\\nChoices: {question_data['choices']}"
        })

        # Get response from Claude
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            messages=messages
        )

        # Add response to messages
        messages.append({
            "role": "assistant",
            "content": response.content
        })

        # Save this trajectory
        logger.log_trajectory(idx, messages)

    # Finalize logging
    logger.finalize(len(questions))

This creates:
    working_dir/agent_execution/execution_q0.json
    working_dir/agent_execution/execution_q1.json
    ...
    working_dir/agent_execution/execution_q197.json

Instead of a single large:
    working_dir/agent_execution.json

Benefits:
- Each trajectory is isolated and independently parseable
- Easier to debug specific questions
- Better for large datasets (no single huge file)
- Feedback agent can analyze patterns across trajectories
"""