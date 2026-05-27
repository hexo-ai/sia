"""Unit tests for orchestrator helper functions."""

import json

from sia.orchestrator import load_agent_execution


def test_load_single_trajectory(tmp_path):
    trajectory = [{"role": "user", "content": "hello"}]
    (tmp_path / "agent_execution.json").write_text(json.dumps(trajectory))

    data, is_multi = load_agent_execution(str(tmp_path))
    assert not is_multi
    assert isinstance(data, list)
    assert data[0]["role"] == "user"


def test_load_multi_trajectory(tmp_path):
    exec_dir = tmp_path / "agent_execution"
    exec_dir.mkdir()

    for i in range(3):
        traj = [{"role": "user", "content": f"question {i}"}]
        (exec_dir / f"execution_q{i}.json").write_text(json.dumps(traj))

    data, is_multi = load_agent_execution(str(tmp_path))
    assert is_multi
    assert data["count"] == 3
    assert len(data["trajectories"]) == 3


def test_load_missing_execution(tmp_path):
    data, _is_multi = load_agent_execution(str(tmp_path))
    assert "error" in data


def test_load_malformed_json(tmp_path):
    (tmp_path / "agent_execution.json").write_text("{not valid json")

    data, is_multi = load_agent_execution(str(tmp_path))
    assert not is_multi
    assert "error" in data or "raw_preview" in data


def test_load_empty_multi_trajectory_folder(tmp_path):
    (tmp_path / "agent_execution").mkdir()

    data, is_multi = load_agent_execution(str(tmp_path))
    assert is_multi
    assert "error" in data
