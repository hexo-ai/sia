"""
call_task_model.py — unified LLM caller for target agents.

Supports:
  - tinker://...       → Tinker SDK + openai_harmony (raw token sampling)
  - openai/gpt-oss-*   → OpenAI client at Tinker endpoint (TINKER_API_KEY preferred,
                         falls back to OPENAI_API_KEY). Response is raw Harmony text;
                         parsed via openai_harmony the same way as the SDK path.
  - anything else      → litellm (Anthropic, Gemini, standard OpenAI, etc.)

Environment variables:
  TINKER_API_KEY    — API key for the Tinker endpoint (gpt-oss models)
  TINKER_BASE_URL   — Override Tinker endpoint (default: https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1)
  OPENAI_API_KEY    — Fallback for gpt-oss when TINKER_API_KEY is absent; also used for openai/* models via litellm
  ANTHROPIC_API_KEY — Required for claude/* models via litellm
  GEMINI_API_KEY    — Required for gemini/* models via litellm (GOOGLE_API_KEY also accepted)
  (litellm auto-selects the right key from the model string prefix for any other provider)

Usage:
    import sys
    sys.path.insert(0, shared_dir)
    from call_task_model import call_task_model, is_tinker

    response = call_task_model(
        messages=[
            {"role": "system",    "content": system_prompt},
            {"role": "developer", "content": developer_prompt},  # includes TypeScript tool defs
            {"role": "user",      "content": "Please complete the task."},
        ],
        tools=[{"name": "bash", "description": "...", "parameters": {...}}],
        model="openai/gpt-oss-120b",   # or tinker://... for Tinker SDK path
    )
    # response = {
    #   "content":      str,         # final channel text (or assistant content for litellm)
    #   "tool_calls":   list[dict],  # [{"name": str, "args": dict, "call_id": str|None}]
    #   "raw_messages": list[dict],  # append to message history for next turn
    # }

Message format (unified, used for input history and raw_messages output):
    {"role": "system",    "content": "..."}
    {"role": "developer", "content": "..."}          # merged into system for litellm/OpenAI
    {"role": "user",      "content": "..."}
    {"role": "assistant", "channel": "final",      "content": "..."}
    {"role": "assistant", "channel": "analysis",   "content": "..."}   # CoT
    {"role": "assistant", "channel": "commentary", "content": "...",
                          "recipient": "functions.bash", "content_type": "<|constrain|> json"}
    {"role": "tool_result", "name": "bash", "content": "...", "call_id": "..."}
"""

from __future__ import annotations

import datetime
import json
import os
import time
from typing import Any


# Transient network error class names worth retrying (covers httpx, openai, tinker, stdlib)
_RETRYABLE = frozenset({
    "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout",
    "PoolTimeout", "RemoteProtocolError", "ReadError",
    "APIConnectionError", "APITimeoutError",
    "ServiceUnavailableError",
    "ConnectionError", "TimeoutError", "ConnectionResetError",
    "ConnectionRefusedError", "BrokenPipeError",
})


def _with_retry(fn, max_attempts: int = 4, base_delay: float = 4.0):
    """Call fn(), retrying on transient network errors with exponential backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_attempts or type(e).__name__ not in _RETRYABLE:
                raise
            time.sleep(base_delay * (2 ** (attempt - 1)))


def is_tinker(model: str) -> bool:
    """True for tinker:// checkpoint paths and openai/gpt-oss-* base models (both use Tinker SDK)."""
    m = model.lower()
    return m.startswith("tinker://") or m.startswith("openai/gpt-oss")





def call_task_model(
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict] | None = None,
    log_dir: str | None = None,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Call the model and return a structured response dict.

    Routing:
      tinker://...        → Tinker SDK + openai_harmony, model_path= (fine-tuned checkpoint)
      openai/gpt-oss-*    → Tinker SDK + openai_harmony, base_model= (base model)
      anything else       → litellm
    """
    if is_tinker(model):
        return _call_tinker_harmony(messages, model, tools, log_dir, temperature)
    return _call_litellm(messages, model, tools, log_dir, temperature)


# ── Debug logging ─────────────────────────────────────────────────────────────

def _write_logs(log_dir: str | None, input_text: str, output_text: str) -> None:
    if log_dir is None:
        return
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    with open(os.path.join(log_dir, f"{ts}_input.txt"),  "w", encoding="utf-8") as f:
        f.write(input_text)
    with open(os.path.join(log_dir, f"{ts}_output.txt"), "w", encoding="utf-8") as f:
        f.write(output_text)


# ── Shared Harmony parser ──────────────────────────────────────────────────────

import re as _re

def _parse_harmony_tokens(output_tokens: list[int]) -> dict[str, Any]:
    """Decode Harmony output token IDs to text and parse via regex."""
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Strip trailing stop token (<|return|>=200002, <|call|>=200012)
    if output_tokens and output_tokens[-1] in (200002, 200012):
        output_tokens = output_tokens[:-1]

    text = encoding.decode_utf8(output_tokens)
    return _parse_harmony_text(text)


def _parse_harmony_text(text: str) -> dict[str, Any]:
    """Parse Harmony-format text into a structured response dict.

    Handles:
      - <|channel|>analysis<|message|>CoT<|end|>
      - <|start|>assistant<|channel|>final<|message|>Answer<|return|>
      - <|start|>assistant<|channel|>commentary to=functions.bash <|constrain|>json<|message|>{"cmd":"..."}<|call|>
      - <|start|>assistant to=functions.bash<|channel|>commentary ...<|message|>...<|call|>  (alt format)
    """
    tool_calls   : list[dict] = []
    raw_messages : list[dict] = []
    final_content: str        = ""

    # Strip leading <|start|>assistant if present (Tinker prompt ends with it;
    # gpt-oss API response may include it too)
    text = _re.sub(r"^<\|start\|>assistant\s*", "", text)

    # Split into per-message segments on <|start|>
    parts = _re.split(r"<\|start\|>", text)

    for i, part in enumerate(parts):
        if not part.strip():
            continue

        recipient    : str | None = None
        content_type : str | None = None

        if i == 0:
            role = "assistant"
        else:
            # Parse role and optional "to=recipient" from the role section
            m = _re.match(r"(\S+?)(?:\s+to=([^<\s]+))?(?=<\|)", part)
            if m:
                role      = m.group(1)
                recipient = m.group(2)
                part      = part[m.end():]
            else:
                role = "assistant"

        # Parse channel section: <|channel|>{channel}[ to=recipient][ content_type]<|message|>
        ch_m = _re.match(r"<\|channel\|>(.*?)<\|message\|>", part, _re.DOTALL)
        if ch_m:
            ch_spec = ch_m.group(1)
            # Extract "to=recipient" from channel spec if present
            to_m = _re.search(r"\bto=(\S+)", ch_spec)
            if to_m and not recipient:
                recipient = to_m.group(1)
                ch_spec = ch_spec[:to_m.start()] + ch_spec[to_m.end():]
            ch_spec = ch_spec.strip()
            # First word is the channel name; rest is content_type
            ch_parts = ch_spec.split(None, 1)
            channel      = ch_parts[0] if ch_parts else ""
            content_type = ch_parts[1].strip() if len(ch_parts) > 1 else None
            content      = part[ch_m.end():]
        else:
            # Fallback: no channel header found — treat everything after <|message|> as content
            msg_m = _re.search(r"<\|message\|>", part)
            channel = ""
            content = part[msg_m.end():] if msg_m else part

        # Strip trailing stop tokens and anything after
        content = _re.sub(r"<\|(?:end|call|return)\|>.*", "", content, flags=_re.DOTALL)

        raw: dict[str, Any] = {"role": role, "channel": channel, "content": content}
        if recipient:
            raw["recipient"] = recipient
        if content_type:
            raw["content_type"] = content_type
        raw_messages.append(raw)

        if channel == "final":
            final_content = content
        elif recipient and recipient.startswith("functions."):
            try:
                args = json.loads(content)
            except Exception:
                args = {"raw": content}
            tool_calls.append({
                "name":    recipient[len("functions."):],
                "args":    args,
                "call_id": None,
            })

    return {"content": final_content, "tool_calls": tool_calls, "raw_messages": raw_messages}


# ── Tinker SDK path (tinker://...) ────────────────────────────────────────────

def _call_tinker_harmony(
    messages: list[dict],
    model: str,
    tools: list[dict] | None,
    log_dir: str | None,
    temperature: float = 0.7,
) -> dict:
    import tinker
    from tinker import types
    from openai_harmony import (
        Author, Conversation, HarmonyEncodingName, Message, Role,
        load_harmony_encoding,
    )

    encoding     = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    harmony_msgs = _build_harmony_messages(messages)
    convo        = Conversation.from_messages(harmony_msgs)
    input_tokens = list(encoding.render_conversation_for_completion(convo, Role.ASSISTANT))

    if model.lower().startswith("tinker://"):
        sampler = tinker.ServiceClient().create_sampling_client(model_path=model)
    else:
        # openai/gpt-oss-* base model
        sampler = tinker.ServiceClient().create_sampling_client(base_model=model)
    model_input = types.ModelInput.from_ints(input_tokens)
    out         = _with_retry(lambda: sampler.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=types.SamplingParams(
            temperature=temperature,
            stop=[200002, 200012],  # <|return|>, <|call|>
        ),
    ).result())

    n_prompt   = len(input_tokens)
    all_tokens = list(out.sequences[0].tokens)
    # Without stop tokens the SDK returns prompt+completion concatenated;
    # with stop tokens configured it returns completion only.
    output_tokens = all_tokens[n_prompt:] if len(all_tokens) > n_prompt else all_tokens

    input_text  = encoding.decode_utf8(input_tokens)
    output_text = encoding.decode_utf8(output_tokens)
    _write_logs(log_dir, input_text, output_text)

    return _parse_harmony_tokens(output_tokens)


# ── litellm path ───────────────────────────────────────────────────────────────

def _call_litellm(
    messages: list[dict],
    model: str,
    tools: list[dict] | None,
    log_dir: str | None,
    temperature: float = 0.7,
) -> dict:
    import litellm

    litellm_msgs = _to_openai_messages(messages)

    kwargs: dict[str, Any] = {
        "model":       model,
        "messages":    litellm_msgs,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = [
            {"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["parameters"],
            }}
            for t in tools
        ]
        kwargs["tool_choice"] = "auto"

    resp   = _with_retry(lambda: litellm.completion(**kwargs))
    lm_msg = resp.choices[0].message

    tool_calls: list[dict] = []
    for tc in (lm_msg.tool_calls or []):
        try:
            args = json.loads(tc.function.arguments)
        except Exception:
            args = {}
        tool_calls.append({"name": tc.function.name, "args": args, "call_id": tc.id})

    content = lm_msg.content or ""
    raw: dict[str, Any] = {"role": "assistant", "channel": None, "content": content}
    if lm_msg.tool_calls:
        raw["tool_calls"] = lm_msg.tool_calls

    output_text = content
    if lm_msg.tool_calls:
        tc_log = [{"name": tc["name"], "args": tc["args"]} for tc in tool_calls]
        output_text = (content + "\n\n" if content else "") + json.dumps(tc_log, indent=2, ensure_ascii=False)
    _write_logs(log_dir,
                json.dumps(litellm_msgs, indent=2, ensure_ascii=False),
                output_text)

    return {"content": content, "tool_calls": tool_calls, "raw_messages": [raw]}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_harmony_messages(messages: list[dict]) -> list:
    """Convert unified message dicts to openai_harmony Message objects."""
    from openai_harmony import Author, Message, Role

    result = []
    for msg in messages:
        role    = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            result.append(Message.from_role_and_content(Role.SYSTEM, content))
        elif role == "developer":
            result.append(Message.from_role_and_content(Role.DEVELOPER, content))
        elif role == "user":
            result.append(Message.from_role_and_content(Role.USER, content))
        elif role == "assistant":
            channel      = msg.get("channel", "final")
            recipient    = msg.get("recipient")
            content_type = msg.get("content_type")
            m = Message.from_role_and_content(Role.ASSISTANT, content).with_channel(channel)
            if recipient:
                m = m.with_recipient(recipient)
            if content_type:
                m = m.with_content_type(content_type)
            result.append(m)
        elif role == "tool_result":
            result.append(
                Message.from_author_and_content(
                    Author.new(Role.TOOL, f"functions.{msg['name']}"),
                    content,
                ).with_channel("commentary")
            )
    return result


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """Convert unified message dicts to plain OpenAI/litellm format."""
    result: list[dict] = []
    for msg in messages:
        role    = msg["role"]
        content = msg.get("content", "")

        if role == "developer":
            if result and result[-1]["role"] == "system":
                result[-1]["content"] += "\n\n" + content
            else:
                result.append({"role": "system", "content": content})
        elif role == "assistant":
            m: dict[str, Any] = {"role": "assistant", "content": content}
            if msg.get("tool_calls"):
                m["tool_calls"] = msg["tool_calls"]
            result.append(m)
        elif role == "tool_result":
            result.append({
                "role":         "tool",
                "tool_call_id": msg.get("call_id") or "unknown",
                "content":      content,
            })
        elif role in ("system", "user"):
            result.append({"role": role, "content": content})
        # skip analysis/commentary — Harmony-specific

    return result
