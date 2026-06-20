import os
import pytest
from sia.agent_impls.pydantic_ai import _make_tools


def test_toolset_sizes(tmp_path):
    assert len(_make_tools(str(tmp_path), "minimal")) == 1
    assert len(_make_tools(str(tmp_path), "standard")) == 3
    assert len(_make_tools(str(tmp_path), "overloaded")) == 9
    assert len(_make_tools(str(tmp_path))) == 3  # default unchanged


def test_minimal_is_write_only(tmp_path):
    tools = _make_tools(str(tmp_path), "minimal")
    assert tools[0].__name__ == "write_file"


def test_decoy_tools_are_noops(tmp_path):
    tools = {t.__name__: t for t in _make_tools(str(tmp_path), "overloaded")}
    assert tools["web_search"]("anything") == "[no results]"
    assert tools["sql_query"]("select 1") == "[no rows]"
    # decoy writes nothing to disk
    before = set(os.listdir(tmp_path))
    tools["calculator"]("2+2")
    assert set(os.listdir(tmp_path)) == before


# ---- context knob ----
from sia.prompts import build_meta_prompt
from sia.run_setup import TaskFiles


def _tf():
    return TaskFiles(
        sample_task_descriptions="SAMPLE_DESC_MARKER",
        reference_target_agent_py="REFERENCE_SEED_MARKER",
        sample_agent_execution={"trajectory": "SAMPLE_TRAJ_MARKER"},
        task_md="TASK_SPEC_MARKER",
    )


def test_context_standard_has_reference_and_trajectory(monkeypatch):
    monkeypatch.delenv("SIA_CONTEXT", raising=False)
    p = build_meta_prompt(_tf(), "m", "/wd")
    assert "REFERENCE_SEED_MARKER" in p
    assert "SAMPLE_TRAJ_MARKER" in p
    assert "ADDITIONAL BACKGROUND" not in p


def test_context_lean_strips_reference_and_trajectory(monkeypatch):
    monkeypatch.setenv("SIA_CONTEXT", "lean")
    p = build_meta_prompt(_tf(), "m", "/wd")
    assert "REFERENCE_SEED_MARKER" not in p
    assert "SAMPLE_TRAJ_MARKER" not in p
    assert "reference omitted" in p
    assert "TASK_SPEC_MARKER" in p  # task spec still present


def test_context_distractor_injects_irrelevant_block(monkeypatch):
    monkeypatch.setenv("SIA_CONTEXT", "distractor")
    p = build_meta_prompt(_tf(), "m", "/wd")
    assert "ADDITIONAL BACKGROUND" in p
    assert "RLHF" in p
    assert "REFERENCE_SEED_MARKER" in p  # still has standard content
