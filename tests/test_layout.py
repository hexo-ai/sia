"""Tests for venv executable path resolution in sia.layout.

These guard cross-platform behavior: a venv puts executables in ``bin/`` on
POSIX and ``Scripts/`` (with ``.exe`` suffixes) on Windows. ``os.name`` is
monkeypatched so both layouts are asserted regardless of the host OS.
"""

import os

import pytest

from sia.layout import venv_pip_path, venv_python_path

# (os.name, bin dir, python exe, pip exe)
_PLATFORMS = [
    ("posix", "bin", "python", "pip"),
    ("nt", "Scripts", "python.exe", "pip.exe"),
]


@pytest.mark.parametrize(("osname", "bindir", "py", "pip"), _PLATFORMS)
def test_venv_python_path(monkeypatch, osname, bindir, py, pip):
    monkeypatch.setattr("sia.layout.os.name", osname)
    # os.path.join in both the helper and here uses the host separator, so the
    # comparison stays valid no matter which OS the test runs on.
    assert venv_python_path("/fake/venv") == os.path.join("/fake/venv", bindir, py)


@pytest.mark.parametrize(("osname", "bindir", "py", "pip"), _PLATFORMS)
def test_venv_pip_path(monkeypatch, osname, bindir, py, pip):
    monkeypatch.setattr("sia.layout.os.name", osname)
    assert venv_pip_path("/fake/venv") == os.path.join("/fake/venv", bindir, pip)


def test_posix_paths_unchanged(monkeypatch):
    """POSIX output must stay byte-identical to the pre-fix behavior."""
    monkeypatch.setattr("sia.layout.os.name", "posix")
    assert venv_python_path("/fake/venv") == os.path.join("/fake/venv", "bin", "python")
    assert venv_pip_path("/fake/venv") == os.path.join("/fake/venv", "bin", "pip")
