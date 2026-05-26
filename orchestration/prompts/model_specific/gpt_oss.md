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
| `<|return|>` | 200002 | Final answer, stop inference |
| `<|call|>`   | 200012 | Tool call, stop inference    |

### Context window
32 768 tokens total (prompt + completion).

### General behavioral tendency — prompt defensively for the main task
gpt-oss-120b tends to explore the filesystem (ls, find, cat) before writing any code,
even when all necessary paths are already provided in the prompt. Left unprompted,
it can burn most of its turn budget on exploration without producing a solution.

The developer message sent to gpt-oss-120b for the main task MUST explicitly counter this:
- State the dataset path, working directory, and evaluate command upfront — leave nothing to discover
- Add a direct instruction: "Write a working solution immediately in turn 1. Do NOT explore the filesystem first."
- Remind it of the turn and time budget so it prioritizes action over investigation

### Calling gpt-oss-120b for specific bounded sub-tasks from the scaffold

gpt-oss-120b is perfectly capable of handling large, open-ended tasks with many turns —
that is its normal operating mode. The following guidance applies **only when the scaffold
calls it for a specific, bounded sub-task** where the expected output is well-defined
(e.g., "give me the column names of this dataset", "generate a solution using approach X").

For these focused sub-calls, be highly directive:

1. **Specify the exact tool call and output format expected.**
   gpt-oss-120b will default to exploration or free-form text unless you tell it
   precisely what to do. For structured output, require a `write_file` call:

       "Use the write_file tool to write the result to <working_dir>/output.json.
        Do NOT write to any other path. Do NOT produce a final-channel text response."

2. **For code generation sub-tasks, require write_file with the target path.**
   Do not ask for code in the final channel — it will be truncated and hard to parse.
   Instead: "Write the complete Python function to <working_dir>/solution.py using write_file."

3. **For analytical sub-tasks (e.g. dataset inspection), require a single write_file.**
   Example prompt for dataset schema extraction:
       "Read the file at <dataset_dir>/train.csv. Write a JSON summary of its schema
        (column names, dtypes, shape, first 3 rows) to <working_dir>/schema.json
        using write_file. Use a single tool call. Do not explore other files."

4. **Allocate a small sub-budget (3–8 turns) for focused sub-tasks.**
   gpt-oss-120b with a large turn budget will over-explore. A tight budget forces
   it to act immediately. Pass max_turns=N where N matches the task complexity.

5. **Never ask for open-ended reasoning in a focused sub-call.**
   "Think about the best approach and implement it" → bad for a sub-task (wastes turns).
   "Implement a Gaussian-weighted kNN denoiser and write it to solution.py" → good.
