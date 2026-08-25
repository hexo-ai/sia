"""Tests for the JSON-defined provider registry."""

import json

import pytest

from sia.providers import Provider, available_providers, load_provider


def test_bundled_providers_present():
    assert set(available_providers()) >= {"anthropic", "gemini", "openai", "together", "nebius"}


def test_load_anthropic_provider():
    p = load_provider("anthropic")
    assert isinstance(p, Provider)
    assert p.provider_id == "anthropic"
    assert p.client_kind == "anthropic"
    assert p.base_url is None
    assert p.api_key_env == "ANTHROPIC_API_KEY"


def test_load_nebius_provider():
    p = load_provider("nebius")
    assert p.client_kind == "openai"
    assert p.base_url == "https://api.tokenfactory.us-central1.nebius.com/v1/"
    assert p.api_key_env == "NEBIUS_API_KEY"


def test_unknown_provider_name_raises():
    with pytest.raises(SystemExit):
        load_provider("does-not-exist")


def test_load_provider_from_path(tmp_path):
    path = tmp_path / "custom.json"
    path.write_text(
        json.dumps(
            {
                "provider_id": "custom",
                "name": "Custom",
                "client_kind": "openai",
                "base_url": "https://x/v1",
                "api_key_env": "X_KEY",
            }
        )
    )
    p = load_provider(str(path))
    assert p.provider_id == "custom"
    assert p.base_url == "https://x/v1"


def test_missing_path_raises():
    with pytest.raises(SystemExit):
        load_provider("/no/such/provider.json")


def test_invalid_client_kind_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"provider_id": "bad", "name": "Bad", "client_kind": "mystery", "api_key_env": "K"}))
    with pytest.raises(SystemExit):
        load_provider(str(path))


def test_user_dir_overrides_bundled(tmp_path, monkeypatch):
    providers_dir = tmp_path / "providers"
    providers_dir.mkdir()
    (providers_dir / "nebius.json").write_text(
        json.dumps(
            {
                "provider_id": "nebius",
                "name": "nebius",
                "client_kind": "openai",
                "base_url": "https://override/v1",
                "api_key_env": "NEBIUS_API_KEY",
            }
        )
    )
    monkeypatch.setenv("SIA_PROVIDERS_DIR", str(providers_dir))
    assert load_provider("nebius").base_url == "https://override/v1"


def test_litellm_prefix_is_optional_and_defaults_to_none(tmp_path):
    """Providers omitting litellm_prefix keep the generic openai route."""
    path = tmp_path / "custom.json"
    path.write_text(
        json.dumps(
            {
                "provider_id": "custom",
                "name": "Custom",
                "client_kind": "openai",
                "base_url": "https://example.test/v1",
                "api_key_env": "CUSTOM_API_KEY",
            }
        ),
        encoding="utf-8",
    )
    assert load_provider(str(path)).litellm_prefix is None
    # The bundled OpenRouter provider names its own litellm provider.
    assert load_provider("openrouter").litellm_prefix == "openrouter"
