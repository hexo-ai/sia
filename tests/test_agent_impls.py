"""Tests for the agent-impl registry and the PydanticAI agent impl."""

import asyncio
import json

import pytest

from sia.agent_impls import available_agent_impls, get_agent_impl


def test_registry_lists_builtin_agent_impls():
    assert set(available_agent_impls()) >= {"claude", "openhands", "pydantic-ai"}


def test_get_agent_impl_returns_callable():
    assert callable(get_agent_impl("claude"))
    assert callable(get_agent_impl("pydantic-ai"))


def test_get_agent_impl_unknown_raises():
    with pytest.raises(ValueError):
        get_agent_impl("does-not-exist")


def test_util_reexports_registry_run_agent():
    from sia.agent_impls import run_agent as impl_run_agent
    from sia.util import run_agent as util_run_agent

    assert util_run_agent is impl_run_agent


def test_pydantic_ai_impl_runs_with_test_model(tmp_path):
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    from sia.agent_impls.pydantic_ai import run_agent_pydantic_ai

    # TestModel drives the agent without network; it exercises each registered tool,
    # so write_file should create a file in the working directory.
    asyncio.run(
        run_agent_pydantic_ai(
            TestModel(),
            "5",
            "Create a file with some content using the write_file tool.",
            str(tmp_path),
        )
    )
    assert any(tmp_path.iterdir())


def test_pydantic_ai_model_passthrough():
    from sia.agent_impls.pydantic_ai import _resolve_model

    # Model specs are passed through unchanged to PydanticAI's native parsing.
    assert _resolve_model("openai:gpt-4o") == "openai:gpt-4o"
    assert _resolve_model("anthropic:claude-sonnet-4-5") == "anthropic:claude-sonnet-4-5"
    # No provider -> still a plain passthrough.
    assert _resolve_model("openai:gpt-4o", None) == "openai:gpt-4o"


def test_pydantic_ai_openrouter_uses_native_provider(monkeypatch):
    pytest.importorskip("pydantic_ai")
    from sia.agent_impls.pydantic_ai import _resolve_model
    from sia.providers import Provider

    monkeypatch.setenv("OPENROUTER_TEST_KEY", "test-key")
    provider = Provider(
        provider_id="openrouter",
        name="OpenRouter",
        client_kind="openai",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_TEST_KEY",
    )

    model = _resolve_model("z-ai/glm-5.2", provider)
    assert model.system == "openrouter"
    assert model.base_url.rstrip("/") == "https://openrouter.ai/api/v1"


def test_pydantic_ai_openai_compatible_provider_requires_configured_api_key(monkeypatch):
    from sia.agent_impls.pydantic_ai import _resolve_model
    from sia.providers import Provider

    monkeypatch.delenv("SIA_MISSING_TEST_KEY", raising=False)
    provider = Provider(
        provider_id="custom",
        name="Custom",
        client_kind="openai",
        base_url="https://example.test/v1",
        api_key_env="SIA_MISSING_TEST_KEY",
    )

    with pytest.raises(RuntimeError, match="SIA_MISSING_TEST_KEY"):
        _resolve_model("custom/model", provider)


def test_pydantic_ai_impl_wraps_malformed_provider_json(tmp_path, monkeypatch):
    pytest.importorskip("pydantic_ai")
    import pydantic_ai

    from sia.agent_impls.pydantic_ai import MalformedProviderResponseError, run_agent_pydantic_ai
    from sia.providers import Provider

    class BoomAgent:
        def __init__(self, model, tools):
            self.model = model
            self.tools = tools

        async def run(self, prompt, usage_limits):
            raise json.JSONDecodeError("Expecting value", "not-json", 0)

    monkeypatch.setattr(pydantic_ai, "Agent", BoomAgent)
    monkeypatch.setenv("OPENROUTER_TEST_KEY", "test-key")
    provider = Provider(
        provider_id="openrouter",
        name="OpenRouter",
        client_kind="openai",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_TEST_KEY",
    )

    with pytest.raises(MalformedProviderResponseError) as exc_info:
        asyncio.run(run_agent_pydantic_ai("z-ai/glm-5.2", "5", "prompt", str(tmp_path), provider=provider))

    message = str(exc_info.value)
    assert "Provider returned a 200 response body" in message
    assert "z-ai/glm-5.2" in message
    assert "OpenRouter" in message
    assert "$OPENROUTER_TEST_KEY" in message


def test_openhands_model_gets_openai_prefix_for_compatible_provider():
    """An OpenAI-compatible provider (base_url) gets an explicit litellm 'openai/' prefix."""
    from sia.agent_impls.openhands import _resolve_model
    from sia.providers import load_provider

    nebius = load_provider("nebius")  # client_kind=openai, has base_url
    assert _resolve_model("moonshotai/Kimi-K2.6", nebius) == "openai/moonshotai/Kimi-K2.6"
    # Already prefixed -> not double-prefixed.
    assert _resolve_model("openai/gpt-4o", nebius) == "openai/gpt-4o"


def test_openhands_model_passthrough_without_compatible_provider():
    """Native (anthropic) and provider-less specs pass through unchanged."""
    from sia.agent_impls.openhands import _resolve_model
    from sia.providers import load_provider

    assert _resolve_model("claude-sonnet-4-5", None) == "claude-sonnet-4-5"
    anthropic = load_provider("anthropic")  # client_kind=anthropic, no base_url
    assert _resolve_model("claude-sonnet-4-5", anthropic) == "claude-sonnet-4-5"


def test_run_agent_threads_provider_to_agent_impl():
    """run_agent forwards the optional provider kwarg to the dispatched agent impl."""
    import asyncio

    from sia.agent_impls import base
    from sia.providers import load_provider

    captured = {}

    async def fake_runner(model, max_turns, prompt, cwd, provider=None):
        captured["provider"] = provider

    base.register("capture-test", fake_runner)
    nebius = load_provider("nebius")
    asyncio.run(base.run_agent("m", "5", "p", "/tmp", agent_impl="capture-test", provider=nebius))
    assert captured["provider"] is nebius


def test_openhands_model_uses_provider_declared_litellm_prefix():
    """A provider naming its own litellm provider is routed there, not via the generic openai one."""
    from sia.agent_impls.openhands import _resolve_model
    from sia.providers import load_provider

    openrouter = load_provider("openrouter")
    assert openrouter.litellm_prefix == "openrouter"
    assert _resolve_model("anthropic/claude-haiku-4.5", openrouter) == "openrouter/anthropic/claude-haiku-4.5"
    # Already-prefixed specs are not double-prefixed.
    assert (
        _resolve_model("openrouter/anthropic/claude-haiku-4.5", openrouter) == "openrouter/anthropic/claude-haiku-4.5"
    )
    # A gateway model id that merely starts with "openai/" is a vendor namespace, not a route:
    # it must still be prefixed (the old hardcoded guard skipped it and left it misrouted).
    assert _resolve_model("openai/gpt-oss-120b", openrouter) == "openrouter/openai/gpt-oss-120b"

    # Providers that declare no prefix keep the generic openai route, unchanged.
    nebius = load_provider("nebius")
    assert nebius.litellm_prefix is None
    assert _resolve_model("moonshotai/Kimi-K2.6", nebius) == "openai/moonshotai/Kimi-K2.6"


def test_run_agent_forwards_model_canonical_name_only_when_set():
    """The canonical name reaches the impl when set, and is omitted otherwise.

    Omitting it keeps runners registered against the older signature -- including third-party
    impls -- working, since ``register()`` is a public extension point.
    """
    import asyncio

    from sia.agent_impls import base

    captured = {}

    async def canonical_runner(model, max_turns, prompt, cwd, provider=None, model_canonical_name=None):
        captured["canonical"] = model_canonical_name

    base.register("canonical-test", canonical_runner)
    asyncio.run(
        base.run_agent("m", "5", "p", "/tmp", agent_impl="canonical-test", model_canonical_name="claude-haiku-4-5")
    )
    assert captured["canonical"] == "claude-haiku-4-5"

    # Unset -> the kwarg is not passed at all, so a legacy runner signature still accepts the call.
    legacy = {}

    async def legacy_runner(model, max_turns, prompt, cwd, provider=None):
        legacy["called"] = True

    base.register("legacy-test", legacy_runner)
    asyncio.run(base.run_agent("m", "5", "p", "/tmp", agent_impl="legacy-test"))
    assert legacy["called"] is True


def test_openrouter_prefix_preserves_cache_control_on_the_wire(monkeypatch):
    """The generic openai route silently strips cache_control; the openrouter route keeps it.

    This is the whole point of provider-declared prefixes: OpenHands emits the breakpoints either
    way, so a capability flag alone proves nothing -- only the serialised body does.
    """
    pytest.importorskip("openhands")
    import json

    import httpx
    from openhands.sdk import LLM
    from openhands.sdk.llm import Message, TextContent

    captured = {}

    def fake_send(self, request, **kwargs):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "anthropic/claude-haiku-4.5",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    monkeypatch.setattr(httpx.Client, "send", fake_send)

    def capture(model_spec):
        captured.clear()
        llm = LLM(
            model=model_spec,
            model_canonical_name="claude-haiku-4-5",
            api_key="sk-not-a-real-key",
            base_url="https://openrouter.ai/api/v1",
            reasoning_effort=None,
            usage_id=model_spec,
        )
        messages = [
            Message(role="system", content=[TextContent(text="SYS")]),
            Message(role="user", content=[TextContent(text="hi")]),
        ]
        llm.completion(messages=messages)
        return llm, captured["body"]

    llm, body = capture("openrouter/anthropic/claude-haiku-4.5")
    assert "cache_control" in json.dumps(body["messages"]), "cache_control must survive to the wire"
    # reasoning_effort is pinned off, so naming a litellm provider cannot silently enable thinking.
    assert "reasoning_effort" not in body
    # Capability lookup via the canonical name also repairs context-window detection.
    assert llm.max_input_tokens and llm.max_output_tokens

    # The generic openai route strips the markers, which is the bug being fixed. Note the
    # capability gate reports caching active in BOTH cases -- only the payload differs.
    _, generic_body = capture("openai/anthropic/claude-haiku-4.5")
    assert "cache_control" not in json.dumps(generic_body["messages"])
