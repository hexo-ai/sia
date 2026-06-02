"""Unit tests for sia.config.Config."""

from dataclasses import fields

from sia.config import Config


def test_default_values():
    cfg = Config()
    assert cfg.DEFAULT_MAX_GENERATIONS == 3
    assert cfg.DEFAULT_BACKEND == "claude"
    assert cfg.SANDBOX_MODE == "none"
    assert cfg.DEFAULT_MAX_TURNS == 20
    assert cfg.DOCKER_MEMORY_LIMIT == "2g"
    assert cfg.MAX_CONTEXT_FILE_SIZE == 10_000_000


def test_from_env_reads_sia_vars(monkeypatch):
    monkeypatch.setenv("SIA_MAX_GENERATIONS", "5")
    monkeypatch.setenv("SIA_BACKEND", "openhands")
    monkeypatch.setenv("SIA_SANDBOX_MODE", "docker")
    monkeypatch.setenv("SIA_META_MODEL", "opus")

    cfg = Config.from_env()
    assert cfg.DEFAULT_MAX_GENERATIONS == 5
    assert cfg.DEFAULT_BACKEND == "openhands"
    assert cfg.SANDBOX_MODE == "docker"
    assert cfg.DEFAULT_CLAUDE_META_MODEL == "opus"


def test_from_env_invalid_value_keeps_default(monkeypatch):
    monkeypatch.setenv("SIA_MAX_GENERATIONS", "not-a-number")

    cfg = Config.from_env()
    assert cfg.DEFAULT_MAX_GENERATIONS == 3


def test_from_env_no_vars_returns_defaults():
    cfg = Config.from_env()
    assert cfg.DEFAULT_MAX_GENERATIONS == 3
    assert cfg.DEFAULT_TASK_MODEL == "claude-haiku-4-5-20251001"


def test_config_is_dataclass_with_expected_fields():
    field_names = {f.name for f in fields(Config)}
    expected = {
        "DEFAULT_CLAUDE_META_MODEL",
        "DEFAULT_TASK_MODEL",
        "DEFAULT_MAX_GENERATIONS",
        "DEFAULT_BACKEND",
        "SANDBOX_MODE",
        "DOCKER_IMAGE",
        "MAX_CONTEXT_FILE_SIZE",
    }
    assert expected.issubset(field_names)
