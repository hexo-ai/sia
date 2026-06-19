# tests/test_verified.py
import os
from sia.config import Config


def test_config_verified_defaults():
    cfg = Config()
    assert cfg.BEST_OF_N == 4
    assert cfg.EARLY_STOP_THRESHOLD == 0.78
    assert cfg.VAL_FRACTION == 0.2
    assert cfg.TRIAGE_MODE == "lint"


def test_config_verified_env(monkeypatch):
    monkeypatch.setenv("SIA_BEST_OF_N", "3")
    monkeypatch.setenv("SIA_EARLY_STOP_THRESHOLD", "0.7")
    monkeypatch.setenv("SIA_VAL_FRACTION", "0.25")
    monkeypatch.setenv("SIA_TRIAGE", "off")
    cfg = Config.from_env()
    assert cfg.BEST_OF_N == 3
    assert cfg.EARLY_STOP_THRESHOLD == 0.7
    assert cfg.VAL_FRACTION == 0.25
    assert cfg.TRIAGE_MODE == "off"
