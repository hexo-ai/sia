"""Tests for file size limit enforcement in context_manager."""

import json

from sia.context_manager import _safe_load_json, _safe_read_file


def test_read_file_under_limit(tmp_path):
    f = tmp_path / "small.txt"
    f.write_text("hello world")
    result = _safe_read_file(str(f), max_bytes=1024)
    assert result == "hello world"


def test_read_file_over_limit(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("x" * 2000)
    result = _safe_read_file(str(f), max_bytes=1000)
    assert result is None


def test_read_file_at_exact_limit(tmp_path):
    content = "a" * 1000
    f = tmp_path / "exact.txt"
    f.write_text(content)
    # File size equals limit (not >), should succeed
    result = _safe_read_file(str(f), max_bytes=1000)
    assert result == content


def test_load_json_under_limit(tmp_path):
    f = tmp_path / "data.json"
    f.write_text(json.dumps({"accuracy": 0.95}))
    result = _safe_load_json(str(f), max_bytes=4096)
    assert result == {"accuracy": 0.95}


def test_load_json_over_limit(tmp_path):
    data = {"key": "x" * 5000}
    f = tmp_path / "big.json"
    f.write_text(json.dumps(data))
    result = _safe_load_json(str(f), max_bytes=1000)
    assert result is None


def test_load_json_nonexistent(tmp_path):
    result = _safe_load_json(str(tmp_path / "nope.json"), max_bytes=4096)
    assert result is None


def test_load_json_invalid_json(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("{not valid json")
    result = _safe_load_json(str(f), max_bytes=4096)
    assert result is None


def test_read_file_nonexistent(tmp_path):
    result = _safe_read_file(str(tmp_path / "missing.txt"), max_bytes=4096)
    assert result is None
