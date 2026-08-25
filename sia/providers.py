"""LLM provider registry — JSON-defined endpoints/credentials.

A ``Provider`` describes *how* to talk to a model provider: the SDK family
(``client_kind``), an optional OpenAI-compatible ``base_url``, and the environment
variable holding the API key. Providers are defined in JSON files (bundled under
``sia/defaults/providers/`` and user-extensible via ``$SIA_PROVIDERS_DIR`` or
``./providers``) and referenced **by name** from an agent profile (see ``sia.profiles``).

Adding a provider is dropping a JSON file — no code change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from sia.config_files import available_names, read_config_text

ENV_VAR = "SIA_PROVIDERS_DIR"
SUBDIR = "providers"

# SDK family the generated/meta agent should use to reach the model.
VALID_CLIENT_KINDS = ("anthropic", "openai", "google")


@dataclass(frozen=True)
class Provider:
    """How to reach a model provider's API."""

    provider_id: str  # stable identifier, referenced by a profile's "provider_id"
    name: str  # human-readable display name
    client_kind: str  # "anthropic" | "openai" | "google"
    base_url: str | None  # None for native endpoints; set for OpenAI-compatible providers
    api_key_env: str
    # litellm routing prefix the openhands agent impl should use for this endpoint. Defaults to
    # "openai" (the generic OpenAI-compatible route). Gateways with a first-class litellm provider
    # -- e.g. "openrouter" -- should name it here: litellm then applies that provider's own request
    # transform instead of the generic one, which preserves prompt-cache breakpoints and returns
    # usage/cost data. See sia.agent_impls.openhands._resolve_model.
    litellm_prefix: str | None = None


def available_providers() -> list[str]:
    """Names of all providers discoverable in the bundled + user directories."""
    return available_names(env_var=ENV_VAR, subdir=SUBDIR)


def load_provider(name_or_path: str) -> Provider:
    """Load and validate a provider by bundled/user name or by path to a .json file."""
    text, source = read_config_text(name_or_path, env_var=ENV_VAR, subdir=SUBDIR, kind="provider")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid provider JSON at {source}: {exc}") from exc

    missing = {"provider_id", "name", "client_kind", "api_key_env"} - data.keys()
    if missing:
        raise SystemExit(f"Provider at {source} is missing required keys: {', '.join(sorted(missing))}")

    client_kind = data["client_kind"]
    if client_kind not in VALID_CLIENT_KINDS:
        raise SystemExit(
            f"Provider at {source} has invalid client_kind '{client_kind}'. "
            f"Expected one of: {', '.join(VALID_CLIENT_KINDS)}."
        )

    return Provider(
        provider_id=data["provider_id"],
        name=data["name"],
        client_kind=client_kind,
        base_url=data.get("base_url"),
        api_key_env=data["api_key_env"],
        litellm_prefix=data.get("litellm_prefix"),
    )
