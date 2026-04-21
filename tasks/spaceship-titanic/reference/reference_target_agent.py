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