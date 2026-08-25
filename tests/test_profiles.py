"""Tests for the JSON-defined agent-profile registry."""

import json

import pytest

from sia.profiles import (
    MetaAgentProfile,
    TargetAgentProfile,
    available_profiles,
    load_meta_agent_profile,
    load_target_agent_profile,
)


def test_bundled_profiles_present():
    assert set(available_profiles()) >= {"default-meta", "default-target", "kimi-nebius-target"}


def test_default_meta_profile():
    p = load_meta_agent_profile("default-meta")
    assert isinstance(p, MetaAgentProfile)
    assert p.profile_id == "default-meta"
    assert p.agent_impl == "claude"
    assert p.model == "haiku"
    assert p.provider.provider_id == "anthropic"


def test_default_target_profile_uses_default_reference():
    p = load_target_agent_profile("default-target")
    assert isinstance(p, TargetAgentProfile)
    assert p.agent_reference.kind == "default"
    assert p.model == "claude-haiku-4-5-20251001"
    assert p.provider.client_kind == "anthropic"


def test_kimi_nebius_target_profile_resolves_provider():
    p = load_target_agent_profile("kimi-nebius-target")
    assert p.agent_reference.kind == "default"
    assert p.model == "moonshotai/Kimi-K2.6"
    assert p.provider.provider_id == "nebius"
    assert p.provider.base_url is not None
    assert p.provider.base_url.endswith("nebius.com/v1/")


def test_unknown_profile_raises():
    with pytest.raises(SystemExit):
        load_meta_agent_profile("nope")


def _write_profile(tmp_path, data):
    path = tmp_path / "p.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_invalid_agent_impl_raises(tmp_path):
    path = _write_profile(
        tmp_path, {"profile_id": "p", "name": "p", "agent_impl": "bogus", "model": "m", "provider_id": "anthropic"}
    )
    with pytest.raises(SystemExit):
        load_meta_agent_profile(path)


def test_claude_agent_impl_requires_anthropic_provider(tmp_path):
    path = _write_profile(
        tmp_path, {"profile_id": "p", "name": "p", "agent_impl": "claude", "model": "m", "provider_id": "nebius"}
    )
    with pytest.raises(SystemExit):
        load_meta_agent_profile(path)


def test_openhands_agent_impl_allows_non_anthropic_provider(tmp_path):
    path = _write_profile(
        tmp_path, {"profile_id": "p", "name": "p", "agent_impl": "openhands", "model": "m", "provider_id": "nebius"}
    )
    profile = load_meta_agent_profile(path)
    assert profile.agent_impl == "openhands"
    assert profile.provider.provider_id == "nebius"


def test_target_profile_defaults_reference_when_omitted(tmp_path):
    path = _write_profile(tmp_path, {"profile_id": "p", "name": "p", "model": "m", "provider_id": "anthropic"})
    profile = load_target_agent_profile(path)
    assert profile.agent_reference.kind == "default"


def test_target_profile_file_reference(tmp_path):
    (tmp_path / "my_agent.py").write_text("print('hi')")
    path = _write_profile(
        tmp_path,
        {
            "profile_id": "p",
            "name": "p",
            "model": "m",
            "provider_id": "anthropic",
            "agent_reference": {"source": "./my_agent.py"},
        },
    )
    profile = load_target_agent_profile(path)
    assert profile.agent_reference.kind == "file"
    assert profile.agent_reference.source is not None
    assert profile.agent_reference.source.name == "my_agent.py"


def test_model_canonical_name_is_optional(tmp_path):
    """A meta profile without model_canonical_name loads with None (today's behaviour)."""
    path = _write_profile(
        tmp_path,
        {
            "profile_id": "p",
            "name": "P",
            "agent_impl": "openhands",
            "model": "anthropic/claude-haiku-4.5",
            "provider_id": "openrouter",
        },
    )
    assert load_meta_agent_profile(path).model_canonical_name is None


def test_model_canonical_name_is_read_when_present():
    """The bundled OpenRouter meta profile carries the vendor's canonical id."""
    profile = load_meta_agent_profile("openrouter-meta")
    assert profile.model == "anthropic/claude-haiku-4.5"
    assert profile.model_canonical_name == "claude-haiku-4-5"
