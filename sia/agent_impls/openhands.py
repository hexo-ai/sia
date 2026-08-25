"""OpenHands SDK agent impl."""

from __future__ import annotations

import os
from datetime import datetime

from sia.agent_impls.base import register
from sia.api_keys import resolve_api_key
from sia.logging_setup import get_logger

logger = get_logger(__name__)


def _resolve_model(model_name, provider=None):
    """Resolve the litellm model spec OpenHands' LLM should use.

    litellm derives the provider from the model string's prefix. For an
    OpenAI-compatible endpoint (a provider with ``client_kind == "openai"`` and a
    ``base_url``), the model must carry an explicit routing prefix so litellm routes to
    that ``base_url`` instead of trying to parse the model's own namespace (e.g.
    ``moonshotai/Kimi-K2.6``) as a provider. Already-prefixed and native (anthropic)
    specs pass through unchanged.

    The prefix comes from the provider's ``litellm_prefix`` and defaults to ``openai``,
    the generic OpenAI-compatible route. A gateway that litellm models as a first-class
    provider should name it (``openrouter``) so litellm applies that provider's own
    request transform: the generic OpenAI transform strips ``cache_control`` from every
    message, which silently disables prompt caching.
    """
    if provider is None or not isinstance(model_name, str):
        return model_name
    if provider.client_kind == "openai" and provider.base_url:
        prefix = provider.litellm_prefix or "openai"
        if not model_name.startswith(f"{prefix}/"):
            return f"{prefix}/{model_name}"
    return model_name


async def run_agent_openhands(
    model_name, max_turns, prompt, agent_working_directory, provider=None, model_canonical_name=None
):
    """Run agent using OpenHands SDK"""
    try:
        from openhands.sdk import LLM, Agent, Conversation, Tool
        from openhands.tools.file_editor import FileEditorTool
        from openhands.tools.terminal import TerminalTool
    except ImportError:
        logger.error("OpenHands SDK not installed. Install with: pip install openhands-ai")
        raise

    logger.info(f"Starting OpenHands agent execution with {model_name} model (max turns: {max_turns})")
    logger.debug("=" * 80)
    logger.debug(f"Working directory: {agent_working_directory}")
    logger.debug("=" * 80)

    turn_count = 0
    start_time = datetime.now()

    try:
        # Determine API key + base_url. An explicit provider takes precedence; otherwise
        # infer the key from the model name.
        base_url = provider.base_url if provider else None
        api_key = os.getenv(provider.api_key_env) if provider else resolve_api_key(model_name)
        if not api_key:
            logger.warning(f"No API key found for model {model_name}. Using LLM_API_KEY if available.")
            api_key = os.getenv("LLM_API_KEY")

        # Create LLM instance. litellm needs an explicit provider prefix to route to a
        # custom OpenAI-compatible base_url (see _resolve_model).
        #
        # reasoning_effort is pinned to None deliberately. The SDK defaults it to "high",
        # and litellm only forwards it for providers it reports as reasoning-capable -- so
        # merely naming a gateway's litellm provider would silently switch the meta agent
        # to extended thinking. Opting into that is a separate, measurable decision.
        llm = LLM(
            model=_resolve_model(model_name, provider),
            model_canonical_name=model_canonical_name,
            api_key=api_key,
            base_url=base_url,
            reasoning_effort=None,
        )
        # The LLM model ignores unknown fields, so an SDK predating model_canonical_name
        # would drop it and leave capability lookups (prompt caching, context window, cost)
        # silently degraded rather than failing.
        if model_canonical_name and getattr(llm, "model_canonical_name", None) != model_canonical_name:
            logger.warning(
                f"The installed OpenHands SDK ignored model_canonical_name={model_canonical_name!r}; "
                "prompt caching and context-window detection will be degraded. Upgrade openhands-ai."
            )

        # Create agent with available tools
        agent = Agent(
            llm=llm,
            tools=[
                Tool(name=TerminalTool.name),
                Tool(name=FileEditorTool.name),
            ],
        )

        # Create conversation with workspace and persistence
        # Trajectory will be saved in: agent_working_directory/openhands_trajectory/
        trajectory_dir = os.path.join(agent_working_directory, "openhands_trajectory")

        conversation = Conversation(agent=agent, workspace=agent_working_directory, persistence_dir=trajectory_dir)

        # Send the task prompt
        logger.debug(f"\n{'─' * 80}")
        logger.debug(f"TURN {turn_count + 1}: Sending prompt to agent")
        logger.debug(f"{'─' * 80}")
        conversation.send_message(prompt)

        # Run the agent
        logger.info(f"Running agent (max turns: {max_turns})...")
        logger.debug(f"  → Trajectory will be saved to: {trajectory_dir}")
        result = conversation.run()

        # Log completion
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.debug(f"\n{'=' * 80}")
        logger.debug(f"Final result: {result}")
        logger.debug(f"{'=' * 80}")
        logger.info(f"Execution complete in {elapsed_time:.2f} seconds")
        logger.debug(f"  ✓ Trajectory saved to: {trajectory_dir}")

    except Exception as e:
        logger.error(f"\n{'!' * 80}")
        logger.error(f"ERROR: {e!s}")
        logger.error(f"{'!' * 80}", exc_info=True)
        raise


register("openhands", run_agent_openhands)
