"""Agent profiles — JSON-defined configuration for one agent role.

Two roles, two shapes:

- A **meta/feedback** agent runs *inside* SIA via a registered ``agent_impl``
  (claude / openhands / pydantic-ai) — see :class:`MetaAgentProfile`.
- The **target** agent is generated code SIA never runs as an engine; it is seeded
  from an ``agent_reference`` (the task package's bundled reference, or a user file /
  directory) and iteratively improved — see :class:`TargetAgentProfile`.

Profiles are JSON files (bundled under ``sia/defaults/profiles/`` and user-extensible
via ``$SIA_PROFILES_DIR`` or ``./profiles``). Each references a
:class:`~sia.providers.Provider` by name. Adding a profile is dropping a JSON file —
no code change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sia.agent_impls import available_agent_impls
from sia.agent_reference import AgentReference, parse_agent_reference
from sia.config_files import available_names, read_config_text
from sia.providers import Provider, load_provider

ENV_VAR = "SIA_PROFILES_DIR"
SUBDIR = "profiles"


@dataclass(frozen=True)
class MetaAgentProfile:
    """Full configuration for the meta/feedback agent role."""

    profile_id: str  # stable identifier (also the value passed to --meta-agent-profile)
    name: str  # human-readable display name
    agent_impl: str  # a registered agent impl (claude / openhands / pydantic-ai)
    model: str
    provider: Provider
    # Canonical model id used only for SDK capability lookups (prompt caching, context window,
    # cost), separate from ``model``, which is the routing id the gateway expects. Set this when
    # the gateway's id differs from the vendor's own -- e.g. OpenRouter serves Anthropic's
    # "claude-haiku-4-5" as "anthropic/claude-haiku-4.5", and capability tables key on the former.
    # Change it whenever you change ``model``: a stale value applies the wrong model's
    # capabilities, token limits and pricing.
    model_canonical_name: str | None = None


@dataclass(frozen=True)
class TargetAgentProfile:
    """Full configuration for the target agent role (generated, never run by SIA)."""

    profile_id: str  # stable identifier (also the value passed to --target-agent-profile)
    name: str  # human-readable display name
    model: str
    provider: Provider
    agent_reference: AgentReference  # where the seed code + deps come from


def available_profiles() -> list[str]:
    """Names of all profiles discoverable in the bundled + user directories."""
    return available_names(env_var=ENV_VAR, subdir=SUBDIR)


def _load_json(name_or_path: str) -> tuple[dict, str]:
    text, source = read_config_text(name_or_path, env_var=ENV_VAR, subdir=SUBDIR, kind="profile")
    try:
        return json.loads(text), source
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid profile JSON at {source}: {exc}") from exc


def _require(data: dict, keys: set[str], source: str) -> None:
    missing = keys - data.keys()
    if missing:
        raise SystemExit(f"Profile at {source} is missing required keys: {', '.join(sorted(missing))}")


def _profile_base_dir(source: str) -> str | None:
    """Directory a profile file lives in (for resolving a relative agent_reference)."""
    return None if source.startswith("<bundled>") else str(Path(source).parent)


def load_meta_agent_profile(name_or_path: str) -> MetaAgentProfile:
    """Load and validate a meta-agent profile by bundled/user name or path to a .json file."""
    data, source = _load_json(name_or_path)
    _require(data, {"profile_id", "name", "agent_impl", "model", "provider_id"}, source)

    provider = load_provider(data["provider_id"])
    profile = MetaAgentProfile(
        profile_id=data["profile_id"],
        name=data["name"],
        agent_impl=data["agent_impl"],
        model=data["model"],
        provider=provider,
        model_canonical_name=data.get("model_canonical_name"),
    )
    _validate_meta(profile, source)
    return profile


def load_target_agent_profile(name_or_path: str) -> TargetAgentProfile:
    """Load and validate a target-agent profile by bundled/user name or path to a .json file."""
    data, source = _load_json(name_or_path)
    _require(data, {"profile_id", "name", "model", "provider_id"}, source)

    agent_reference = parse_agent_reference(data.get("agent_reference"), _profile_base_dir(source))
    return TargetAgentProfile(
        profile_id=data["profile_id"],
        name=data["name"],
        model=data["model"],
        provider=load_provider(data["provider_id"]),
        agent_reference=agent_reference,
    )


def _validate_meta(profile: MetaAgentProfile, source: str) -> None:
    """Reject incoherent agent_impl/provider combinations for the meta agent."""
    valid = available_agent_impls()
    if profile.agent_impl not in valid:
        raise SystemExit(
            f"Profile at {source} has invalid agent_impl '{profile.agent_impl}'. Expected one of: {', '.join(valid)}."
        )
    # The Claude Code SDK only talks to Anthropic; pairing it with another provider
    # would silently authenticate against the wrong endpoint.
    if profile.agent_impl == "claude" and profile.provider.client_kind != "anthropic":
        raise SystemExit(
            f"Profile at {source} pairs agent_impl 'claude' with provider "
            f"'{profile.provider.name}' (client_kind={profile.provider.client_kind}). "
            f"The claude agent impl requires an anthropic provider; use the openhands or "
            f"pydantic-ai agent impl for other providers."
        )
