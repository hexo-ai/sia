"""PydanticAI agent impl for running meta/feedback agents.

Builds a PydanticAI Agent with bash + file tools and caps iterations via
UsageLimits. The PydanticAI SDK is imported lazily so the impl can be listed in
the registry even when the optional ``[pydantic-ai]`` extra isn't installed.

The model spec is passed through to PydanticAI's native model parsing (e.g.
"openai:gpt-4o", "anthropic:claude-...", "google-gla:gemini-..."); a non-string
(a PydanticAI Model instance) is used as-is (e.g. TestModel in tests).
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime

from sia.agent_impls.base import register
from sia.config import Config
from sia.logging_setup import get_logger

logger = get_logger(__name__)


def _resolve_model(model_name, provider=None):
    """Resolve the model spec for PydanticAI.

    Without a provider (or for a non-string spec like TestModel) the value is passed
    through to PydanticAI's native parsing. For an OpenAI-compatible provider with a
    base_url, build an OpenAIChatModel pointed at that endpoint.
    """
    if not isinstance(model_name, str) or provider is None:
        return model_name
    if provider.client_kind == "openai" and provider.base_url:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(base_url=provider.base_url, api_key=os.getenv(provider.api_key_env)),
        )
    return model_name


def _make_tools(working_dir: str, toolset: str = "standard"):
    """File + bash tools operating within ``working_dir`` (paths resolve relative to it).

    ``toolset`` selects a harness-knob configuration (audit experiment T):
      - "minimal":    [write_file]                         (it must write the target)
      - "standard":   [write_file, read_file, bash]        (default; unchanged)
      - "overloaded": standard + 6 plausible-but-useless decoy tools, to probe the
                      survey's "tool overload -> tool-selection confusion" claim.
    """

    def _resolve(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(working_dir, path)

    def write_file(path: str, content: str) -> str:
        """Write (overwrite) a file with the given content."""
        target = _resolve(path)
        try:
            parent = os.path.dirname(target)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Written {len(content)} characters to '{target}'."
        except OSError as e:
            return f"Error writing file: {e}"

    def read_file(path: str) -> str:
        """Read and return the contents of a file."""
        target = _resolve(path)
        try:
            with open(target, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: File '{target}' not found."
        except OSError as e:
            return f"Error reading file: {e}"

    def bash(command: str) -> str:
        """Run a bash command in the working directory and return stdout + stderr."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=Config().SHELL_TIMEOUT,
                cwd=working_dir,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out."
        except OSError as e:
            return f"Error running command: {e}"

    # --- decoy tools for the "overloaded" toolset (audit experiment T) ---
    # Plausible, honestly-documented, but useless for writing a self-contained script.
    # No side effects; each returns a short canned string.
    def web_search(query: str) -> str:
        """Search the web for a query and return results."""
        return "[no results]"

    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression."""
        return "[unavailable]"

    def list_directory(path: str) -> str:
        """List the entries in a directory."""
        return "[empty]"

    def http_get(url: str) -> str:
        """Fetch the contents of a URL."""
        return "[no response]"

    def sql_query(query: str) -> str:
        """Run a SQL query against the project database."""
        return "[no rows]"

    def summarize(text: str) -> str:
        """Summarize a block of text."""
        return "[summary unavailable]"

    if toolset == "minimal":
        return [write_file]
    if toolset == "overloaded":
        return [write_file, read_file, bash,
                web_search, calculator, list_directory, http_get, sql_query, summarize]
    return [write_file, read_file, bash]


async def run_agent_pydantic_ai(model_name, max_turns, prompt, agent_working_directory, provider=None):
    """Run a meta/feedback agent using the PydanticAI Agent framework."""
    try:
        from pydantic_ai import Agent
        from pydantic_ai.usage import UsageLimits
    except ImportError:
        logger.error("PydanticAI not installed. Install with: pip install 'sia-agent[pydantic-ai]'")
        raise

    logger.info(f"Starting PydanticAI agent execution with {model_name} model (max turns: {max_turns})")
    logger.debug("=" * 80)
    logger.debug(f"Working directory: {agent_working_directory}")
    logger.debug("=" * 80)

    start_time = datetime.now()

    try:
        request_limit = int(max_turns)
    except (TypeError, ValueError):
        request_limit = Config().DEFAULT_MAX_TURNS

    try:
        toolset = os.environ.get("SIA_TOOLSET", "standard")
        agent = Agent(_resolve_model(model_name, provider),
                      tools=_make_tools(agent_working_directory, toolset))
        result = await agent.run(prompt, usage_limits=UsageLimits(request_limit=request_limit))

        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.debug(f"\n{'=' * 80}")
        logger.debug(f"Final result: {result.output}")
        logger.debug(f"{'=' * 80}")
        logger.info(f"Execution complete in {elapsed_time:.2f} seconds")

    except Exception as e:
        logger.error(f"\n{'!' * 80}")
        logger.error(f"ERROR: {e!s}")
        logger.error(f"{'!' * 80}", exc_info=True)
        raise


register("pydantic-ai", run_agent_pydantic_ai)
