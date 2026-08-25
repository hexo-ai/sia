"""Claude Code SDK agent impl."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, cast

from sia.agent_impls.base import register
from sia.logging_setup import get_logger

logger = get_logger(__name__)

# Tools that take a path or path-glob input; their inputs are resolved against the
# protected dirs. Bash is handled separately (its command is substring-checked).
_FILE_TOOLS = frozenset({"Read", "Edit", "Write", "NotebookEdit", "Glob", "LS", "Grep"})

# Input keys that hold a single path or path-glob across the file tools above.
_PATH_FIELDS = ("file_path", "path", "notebook_path", "pattern")


def _looks_like_path(value: str) -> bool:
    """Heuristic: a string that contains a separator or a parent-dir hop is path-like."""
    return ("/" in value) or value.startswith("..") or value.startswith("~")


def _resolves_inside(candidate: str, protected_real: list[str]) -> bool:
    """True when ``candidate`` (a path, possibly with ``..`` or a glob) resolves inside any
    protected dir. The leading glob magic is stripped so ``<protected>/**`` resolves to the
    protected dir itself rather than a literal ``**`` child.
    """
    cleaned = candidate.split("*", 1)[0].split("?", 1)[0]
    if not cleaned:
        cleaned = candidate
    real = os.path.realpath(os.path.abspath(cleaned))
    return any(real == prot or real.startswith(prot + os.sep) for prot in protected_real)


def _accesses_protected_path(tool_name: str, tool_input: dict, protected_dirs: list[str]) -> bool:
    """Return True when a tool call would touch a protected directory.

    File tools are checked by resolving their path-like input fields (handling ``..`` and
    glob magic) against the canonical protected dirs. Bash is checked by substring match
    of the command against both the canonical absolute protected path and a normalized
    relative form (e.g. ``data/private``). Unknown tools and path-free inputs are allowed.
    """
    if not protected_dirs:
        return False

    protected_real = [os.path.realpath(os.path.abspath(d)) for d in protected_dirs]

    if tool_name == "Bash":
        command = tool_input.get("command", "")
        if not command:
            return False
        relative_forms = [os.path.join("data", os.path.basename(p)) for p in protected_dirs]
        needles = protected_real + relative_forms
        return any(needle in command for needle in needles)

    if tool_name in _FILE_TOOLS:
        candidates: list[str] = []
        for field in _PATH_FIELDS:
            value = tool_input.get(field)
            if isinstance(value, str) and value:
                candidates.append(value)
        # Catch any other path-like string argument (defensive -- tool schemas vary).
        for value in tool_input.values():
            if isinstance(value, str) and value and _looks_like_path(value) and value not in candidates:
                candidates.append(value)
        return any(_resolves_inside(c, protected_real) for c in candidates)

    return False


def _make_protected_path_hook(
    protected_dirs: list[str],
) -> Callable[[dict, str | None, Any], Awaitable[dict]]:
    """Build a PreToolUse hook callback that denies any tool call touching a protected dir.

    Returns the canonical SDK deny dict (``permissionDecision: "deny"``) when the call
    would access a held-out dir, and ``{}`` (allow) otherwise. The reason names the
    held-out dir generically so the agent self-corrects without learning its contents.
    """
    held_out = ", ".join(protected_dirs)

    async def deny_cb(input: dict, tool_use_id: str | None, context: Any) -> dict:
        tool_name = input.get("tool_name", "")
        tool_input = input.get("tool_input", {}) or {}
        if _accesses_protected_path(tool_name, tool_input, protected_dirs):
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"Access to the held-out evaluation directory ({held_out}) is blocked. "
                        "It holds grader-only ground truth; improve the agent from the provided "
                        "feedback alone."
                    ),
                }
            }
        return {}

    return deny_cb


def _build_claude_options(model_name, max_turns, agent_working_directory, protected_paths=None):
    """Construct ClaudeAgentOptions, registering a protected-path deny hook when requested.

    Extracted so the option wiring (notably the PreToolUse hook registration) is unit
    testable without invoking the Claude CLI. When ``protected_paths`` is None/empty no hook
    is registered, so behavior is byte-identical to the pre-hook path.
    """
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

    hooks = None
    if protected_paths:
        deny_cb = _make_protected_path_hook(protected_paths)
        # cast: the SDK's HookCallback is a broad TypedDict-union signature the closure
        # conforms to at runtime but ty cannot match structurally.
        hooks = {"PreToolUse": [HookMatcher(hooks=[cast(Any, deny_cb)])]}

    return ClaudeAgentOptions(
        cwd=agent_working_directory,
        allowed_tools=["Bash", "Read", "Write", "Edit", "Glob"],
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        model=model_name,
        hooks=cast(Any, hooks),
    )


async def run_agent_claude(model_name, max_turns, prompt, agent_working_directory, provider=None, protected_paths=None):
    """Run agent using Claude Code SDK

    The ``provider`` argument is accepted for a uniform agent-impl signature but ignored:
    the Claude Code SDK authenticates against Anthropic natively (ANTHROPIC_API_KEY).

    When ``protected_paths`` is non-empty, a PreToolUse hook denies every tool call that
    would access one of those directories (the task's held-out ground truth). The hook
    fires regardless of ``permission_mode="bypassPermissions"``, so normal authoring is
    unimpeded and only protected-path access is blocked.

    Note: Claude Code automatically saves trajectories to ~/.claude/projects/
    """
    from claude_agent_sdk import ResultMessage, query

    logger.info(f"Starting agent execution with {model_name} model (max turns: {max_turns})")
    logger.debug("=" * 80)
    logger.debug(f"Working directory: {agent_working_directory}")
    logger.debug("=" * 80)

    turn_count = 0
    start_time = datetime.now()

    try:
        async for message in query(
            prompt=prompt,
            options=_build_claude_options(model_name, max_turns, agent_working_directory, protected_paths),
        ):
            logged_content = False

            if hasattr(message, "content") and message.content:
                for block in message.content:
                    # Log agent text responses
                    if hasattr(block, "text") and block.text:
                        if not logged_content:
                            turn_count += 1
                            logger.debug(f"\n{'─' * 80}")
                            logger.debug(f"TURN {turn_count}: Agent Response")
                            logger.debug(f"{'─' * 80}")
                            logged_content = True
                        logger.debug(f"{block.text}")

                    # Log tool calls
                    elif hasattr(block, "name"):
                        if not logged_content:
                            turn_count += 1
                            logger.debug(f"\n{'─' * 80}")
                            logger.debug(f"TURN {turn_count}: Tool Execution")
                            logger.debug(f"{'─' * 80}")
                            logged_content = True

                        logger.debug(f"🔧 Tool: {block.name}")
                        if hasattr(block, "input") and block.input:
                            # Pretty print tool input
                            import json

                            try:
                                input_str = json.dumps(block.input, indent=2)
                                logger.debug(f"   Input: {input_str}")
                            except (TypeError, ValueError):
                                logger.debug(f"   Input: {block.input}")

                    # Log tool results
                    elif hasattr(block, "type") and block.type == "tool_result":
                        if hasattr(block, "content"):
                            result = block.content if isinstance(block.content, str) else str(block.content)
                            # Truncate very long outputs
                            if len(result) > 500:
                                result = result[:500] + f"\n... (truncated, {len(result)} total chars)"
                            logger.debug(f"   ✓ Result: {result}")

            # Log final result
            if isinstance(message, ResultMessage):
                elapsed_time = (datetime.now() - start_time).total_seconds()
                logger.debug(f"\n{'=' * 80}")
                logger.debug(f"Final result: {message.result}")
                logger.debug(f"{'=' * 80}")
                logger.info(f"Execution complete: {turn_count} turns in {elapsed_time:.2f} seconds")

    except Exception as e:
        logger.error(f"\n{'!' * 80}")
        logger.error(f"ERROR: {e!s}")
        logger.error(f"{'!' * 80}", exc_info=True)
        raise


register("claude", run_agent_claude)
