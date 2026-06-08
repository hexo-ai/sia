"""Agent-implementation registry.

An *agent impl* runs a meta/feedback agent: an async runner with the signature
``run(model_name, max_turns, prompt, agent_working_directory)``. Implementations
register themselves by a unique id; the orchestrator and CLI discover them via this
registry, so adding one is a single ``register(...)`` call. Optional SDK imports
happen lazily inside each runner, so the registry lists every impl regardless of
what's installed.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from sia.logging_setup import get_logger

if TYPE_CHECKING:
    from sia.providers import Provider

logger = get_logger(__name__)

# Runners accept (model_name, max_turns, prompt, working_dir) and an optional provider
# (keyword, default None) describing the endpoint/credentials for the meta agent.
AgentRunner = Callable[..., Awaitable[None]]

REGISTRY: dict[str, AgentRunner] = {}


def register(name: str, runner: AgentRunner) -> AgentRunner:
    """Register an agent-impl runner under ``name``."""
    REGISTRY[name] = runner
    return runner


def available_agent_impls() -> list[str]:
    """Ids of all registered agent impls."""
    return list(REGISTRY)


def get_agent_impl(name: str) -> AgentRunner:
    """Return the runner registered under ``name`` (raises ValueError if unknown)."""
    if name not in REGISTRY:
        available = ", ".join(available_agent_impls())
        raise ValueError(f"Unknown agent impl: {name}. Available: {available}")
    return REGISTRY[name]


async def run_agent(
    model_name: str,
    max_turns: str,
    prompt: str,
    agent_working_directory: str,
    agent_impl: str = "claude",
    provider: Provider | None = None,
    protected_paths: list[str] | None = None,
) -> None:
    """Dispatch to the named agent impl.

    Args:
        model_name: The model to use (format depends on the agent impl).
        max_turns: Maximum number of turns for the agent.
        prompt: The task prompt to send to the agent.
        agent_working_directory: Working directory for the agent.
        agent_impl: Which registered impl to use (e.g. "claude", "openhands", "pydantic-ai").
        provider: Optional endpoint/credentials for the model (api_key_env, base_url).
        protected_paths: Held-out dirs the agent must not access. Honored only by the
            "claude" impl (which can enforce a PreToolUse deny hook); other impls ignore it.
    """
    logger.info(f"Using {agent_impl} agent impl")
    runner = get_agent_impl(agent_impl)
    if agent_impl == "claude":
        await runner(
            model_name,
            max_turns,
            prompt,
            agent_working_directory,
            provider=provider,
            protected_paths=protected_paths,
        )
    else:
        await runner(model_name, max_turns, prompt, agent_working_directory, provider=provider)
