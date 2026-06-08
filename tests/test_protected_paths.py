"""Tests for the generic held-out-data seal: the protected-path matcher,
the PreToolUse deny hook, and the ClaudeAgentOptions builder wiring.

All tests are offline -- they exercise the pure matcher and the option-builder directly,
with no live Claude SDK / CLI invocation.
"""

import asyncio
import inspect

import pytest

from sia.agent_impls.claude import (
    _accesses_protected_path,
    _build_claude_options,
    _make_protected_path_hook,
)
from sia.orchestrator import restricted_access

PROTECTED = "/tmp/task/data/private"


# ---------------------------------------------------------------------------
# _accesses_protected_path matrix
# ---------------------------------------------------------------------------


def test_read_inside_protected_is_blocked():
    assert _accesses_protected_path("Read", {"file_path": f"{PROTECTED}/gold.json"}, [PROTECTED])


def test_read_outside_protected_is_allowed():
    assert not _accesses_protected_path("Read", {"file_path": "/tmp/task/cand_0/target_agent.py"}, [PROTECTED])


def test_bash_cat_protected_is_blocked():
    assert _accesses_protected_path("Bash", {"command": f"cat {PROTECTED}/gold.json"}, [PROTECTED])


def test_bash_public_db_is_allowed():
    assert not _accesses_protected_path("Bash", {"command": "sqlite3 data/public/x.db 'select 1'"}, [PROTECTED])


def test_bash_find_protected_is_blocked():
    assert _accesses_protected_path("Bash", {"command": f"find {PROTECTED} -name '*'"}, [PROTECTED])


def test_bash_relative_protected_form_is_blocked():
    # The normalized relative form `data/private` is matched even without the absolute path.
    assert _accesses_protected_path("Bash", {"command": "cd data/private && ls"}, [PROTECTED])


def test_glob_protected_is_blocked():
    assert _accesses_protected_path("Glob", {"pattern": f"{PROTECTED}/**"}, [PROTECTED])


def test_glob_public_is_allowed():
    assert not _accesses_protected_path("Glob", {"pattern": "data/public/*"}, [PROTECTED])


def test_edit_inside_protected_is_blocked():
    assert _accesses_protected_path("Edit", {"file_path": f"{PROTECTED}/gold.json"}, [PROTECTED])


def test_write_inside_protected_is_blocked():
    assert _accesses_protected_path("Write", {"file_path": f"{PROTECTED}/leak.txt"}, [PROTECTED])


def test_relative_dotdot_resolving_to_protected_is_blocked():
    # A `..` path that resolves into the protected dir must be caught after realpath/abspath.
    candidate = f"{PROTECTED}/../private/gold.json"
    assert _accesses_protected_path("Read", {"file_path": candidate}, [PROTECTED])


def test_unknown_tool_is_allowed():
    assert not _accesses_protected_path("WebSearch", {"query": PROTECTED}, [PROTECTED])


def test_empty_protected_dirs_allows_everything():
    assert not _accesses_protected_path("Read", {"file_path": f"{PROTECTED}/gold.json"}, [])


# ---------------------------------------------------------------------------
# Hook output shapes
# ---------------------------------------------------------------------------


def _run_hook(tool_name, tool_input, protected_dirs=None):
    cb = _make_protected_path_hook(protected_dirs or [PROTECTED])
    return asyncio.run(cb({"tool_name": tool_name, "tool_input": tool_input}, None, None))


def test_hook_denies_protected_access_with_canonical_shape():
    result = _run_hook("Read", {"file_path": f"{PROTECTED}/gold.json"})
    output = result["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse"
    assert output["permissionDecision"] == "deny"
    assert PROTECTED in output["permissionDecisionReason"]


def test_hook_allows_non_protected_access_with_empty_dict():
    result = _run_hook("Read", {"file_path": "/tmp/task/cand_0/target_agent.py"})
    assert result == {}


# ---------------------------------------------------------------------------
# ClaudeAgentOptions builder wiring
# ---------------------------------------------------------------------------


def test_builder_registers_hook_when_protected_paths_present():
    options = _build_claude_options("haiku", "20", "/tmp/wd", protected_paths=[PROTECTED])
    assert options.hooks is not None
    matchers = options.hooks["PreToolUse"]
    assert len(matchers) == 1
    assert len(matchers[0].hooks) == 1


def test_builder_registers_no_hook_when_protected_paths_empty():
    options = _build_claude_options("haiku", "20", "/tmp/wd", protected_paths=[])
    assert options.hooks is None


def test_builder_registers_no_hook_when_protected_paths_none():
    options = _build_claude_options("haiku", "20", "/tmp/wd", protected_paths=None)
    assert options.hooks is None


# ---------------------------------------------------------------------------
# Orchestrator wiring: compute_protected_paths
# ---------------------------------------------------------------------------


def test_compute_protected_paths_returns_private_when_present(tmp_path):
    from sia.orchestrator import compute_protected_paths

    private = tmp_path / "data" / "private"
    private.mkdir(parents=True)
    result = compute_protected_paths(str(tmp_path))
    assert result == [str(private.resolve())]


def test_compute_protected_paths_empty_when_absent(tmp_path):
    from sia.orchestrator import compute_protected_paths

    (tmp_path / "data" / "public").mkdir(parents=True)
    assert compute_protected_paths(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# Genericity: the seal code must stay task-agnostic
# ---------------------------------------------------------------------------


def test_seal_code_carries_no_task_specific_identifiers():
    """The seal must reference the held-out dir abstractly, never a concrete task's
    SQL/gold/benchmark vocabulary -- otherwise it leaks task knowledge into a generic layer.
    """
    import sia.agent_impls.claude as claude_mod

    sources = "\n".join(
        inspect.getsource(obj)
        for obj in (
            claude_mod._looks_like_path,
            claude_mod._resolves_inside,
            claude_mod._accesses_protected_path,
            claude_mod._make_protected_path_hook,
            claude_mod._build_claude_options,
        )
    ).lower()

    forbidden = ("sql", "gold", "lawbench", "gpqa", "spaceship", "chess")
    leaked = [token for token in forbidden if token in sources]
    assert not leaked, f"seal code leaked task-specific identifiers: {leaked}"


# ---------------------------------------------------------------------------
# OS-level guard: restricted_access (impl-agnostic backstop)
# ---------------------------------------------------------------------------


def _make_private(tmp_path):
    private = tmp_path / "data" / "private"
    private.mkdir(parents=True)
    (private / "answers.json").write_text("gold")
    return private


def test_restricted_access_strips_then_restores_permissions(tmp_path):
    private = _make_private(tmp_path)
    original = private.stat().st_mode

    with restricted_access([str(private)], enabled=True):
        # All permission bits stripped during the block (root-safe: check the mode, not access).
        assert private.stat().st_mode & 0o777 == 0
    assert private.stat().st_mode == original  # restored after the block


def test_restricted_access_is_noop_when_disabled(tmp_path):
    private = _make_private(tmp_path)
    original = private.stat().st_mode
    with restricted_access([str(private)], enabled=False):
        assert private.stat().st_mode == original  # untouched
    assert private.stat().st_mode == original


def test_restricted_access_restores_on_error(tmp_path):
    private = _make_private(tmp_path)
    original = private.stat().st_mode
    with pytest.raises(ValueError), restricted_access([str(private)], enabled=True):
        raise ValueError("boom")
    assert private.stat().st_mode == original  # restored even on exception


def test_restricted_access_tolerates_missing_path(tmp_path):
    missing = str(tmp_path / "data" / "private")  # never created
    with restricted_access([missing], enabled=True):
        pass  # must not raise
