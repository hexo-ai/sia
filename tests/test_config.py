"""Unit tests for sia.config.Config."""

from dataclasses import fields

from sia.config import Config, _to_bool, _to_str_tuple


def test_default_values():
    cfg = Config()
    assert cfg.DEFAULT_MAX_GENERATIONS == 3
    assert cfg.DEFAULT_AGENT_IMPL == "claude"
    assert cfg.SANDBOX_MODE == "none"
    assert cfg.DEFAULT_MAX_TURNS == 20
    assert cfg.DOCKER_MEMORY_LIMIT == "2g"
    assert cfg.MAX_CONTEXT_FILE_SIZE == 10_000_000


def test_from_env_reads_sia_vars(monkeypatch):
    monkeypatch.setenv("SIA_MAX_GENERATIONS", "5")
    monkeypatch.setenv("SIA_AGENT_IMPL", "openhands")
    monkeypatch.setenv("SIA_SANDBOX_MODE", "docker")
    monkeypatch.setenv("SIA_META_MODEL", "opus")

    cfg = Config.from_env()
    assert cfg.DEFAULT_MAX_GENERATIONS == 5
    assert cfg.DEFAULT_AGENT_IMPL == "openhands"
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
        "DEFAULT_AGENT_IMPL",
        "SANDBOX_MODE",
        "DOCKER_IMAGE",
        "MAX_CONTEXT_FILE_SIZE",
    }
    assert expected.issubset(field_names)


# --- verifier->feedback contract knobs (gold-free eval summary) ---------------


def test_verifier_feedback_knob_defaults():
    cfg = Config()
    assert cfg.VERIFIER_PASS_STATUSES == ("CORRECT", "PASS", "correct")
    assert cfg.FEEDBACK_FAILURE_SAMPLES == 20
    assert cfg.PRIVATE_DIR_GUARD is True


def test_from_env_disables_private_dir_guard(monkeypatch):
    monkeypatch.setenv("SIA_PRIVATE_DIR_GUARD", "0")
    cfg = Config.from_env()
    assert cfg.PRIVATE_DIR_GUARD is False


def test_from_env_overrides_verifier_pass_statuses_tuple(monkeypatch):
    monkeypatch.setenv("SIA_VERIFIER_PASS_STATUSES", "OK, GREEN ,done")
    cfg = Config.from_env()
    assert cfg.VERIFIER_PASS_STATUSES == ("OK", "GREEN", "done")


def test_from_env_overrides_feedback_failure_samples(monkeypatch):
    monkeypatch.setenv("SIA_FEEDBACK_FAILURE_SAMPLES", "5")
    cfg = Config.from_env()
    assert cfg.FEEDBACK_FAILURE_SAMPLES == 5


def test_to_str_tuple_trims_and_drops_empty_parts():
    assert _to_str_tuple("a, b ,, c") == ("a", "b", "c")


def test_to_bool_parses_truthy_and_falsy_strings():
    assert _to_bool("1") is True
    assert _to_bool("TRUE") is True
    assert _to_bool(" yes ") is True
    assert _to_bool("0") is False
    assert _to_bool("false") is False
    assert _to_bool("") is False
    assert _to_str_tuple("") == ()
