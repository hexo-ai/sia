"""
Model-specific guidelines injected into meta-agent and feedback agent prompts.

Add a new elif block when onboarding a new model.
"""


def get_guidelines(model: str) -> str:
    """Return guidelines for the given model string, or empty string if none."""
    m = model.lower()

    if "gpt-oss" in m or "tinker" in m:
        return _GPT_OSS_120B

    return ""


_GPT_OSS_120B = """
## Model-specific guidelines: openai/gpt-oss-120b (Tinker)

### Use the openai_harmony library
gpt-oss-120b uses the Harmony response format. Do NOT write a custom text parser.
Use the `openai_harmony` PyPI package (already installed in the venv):

    from openai_harmony import (
        Author, Conversation, HarmonyEncodingName, Message, Role,
        load_harmony_encoding,
    )
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Encode messages → token IDs
    tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

    # Decode output token IDs → structured messages
    parsed = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)

Alternatively, import from shared_dir:

    from call_task_model import call_task_model, is_tinker

### Message roles
| Role        | Purpose                                                          |
|-------------|------------------------------------------------------------------|
| system      | Identity, date, reasoning level, valid channels                  |
| developer   | Instructions (the "system prompt") + tool definitions            |
| user        | The actual task input                                            |
| assistant   | Model output — tagged with a channel (see below)                 |
| tool_result | Result of a tool call (role = "tool_result", name = "bash", ...) |

### Channels
| Channel     | Purpose                                                          |
|-------------|------------------------------------------------------------------|
| analysis    | Chain-of-thought. Drop from history after final; keep for tool turns. |
| commentary  | Tool calls (to=functions.<name>) and tool results                |
| final       | The answer shown to the user                                     |

### System message format
    You are ChatGPT, a large language model trained by OpenAI.
    Knowledge cutoff: 2024-06
    Current date: {YYYY-MM-DD}

    Reasoning: high

    # Valid channels: analysis, commentary, final. Channel must be included for every message.
    Calls to these tools must go to the commentary channel: 'functions'.

### Tool definitions (developer message)
Define tools in TypeScript namespace syntax inside the developer message:

    # Tools

    ## functions

    namespace functions {

    // description
    type tool_name = (_: {
    // param description
    param: string,
    }) => any;

    } // namespace functions

### Multi-turn tool calling loop
Sample → parse → execute ONE tool call → append result → re-sample → repeat until
the model outputs a final-channel message with no tool calls.
The sampler stops at each `<|call|>` token — one tool call per sampling pass by design.

### CoT handling
- Tool call turn:    keep analysis messages in history (model needs its own reasoning context)
- Final response:    drop analysis messages (conversation is done)

### Stop tokens
| Token       | ID     | Meaning                      |
|-------------|--------|------------------------------|
| <|return|>  | 200002 | Final answer, stop inference |
| <|call|>    | 200012 | Tool call, stop inference    |

### Context window
32 768 tokens total (prompt + completion).

### Known behavioral tendency — prompt defensively
gpt-oss-120b tends to explore the filesystem (ls, find, cat) before writing any code,
even when all necessary paths are already provided in the prompt. Left unprompted,
it can burn most of its turn budget on exploration without producing a solution.

The developer message sent to gpt-oss-120b MUST explicitly counter this:
- State the dataset path, working directory, and evaluate command upfront — leave nothing to discover
- Add a direct instruction: "Write a working solution immediately in turn 1. Do NOT explore the filesystem first."
- Remind it of the turn and time budget so it prioritizes action over investigation
"""
