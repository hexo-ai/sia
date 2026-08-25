"""Unit tests for the Harbor integration (pure logic; no Docker/network)."""

import json

from sia import harbor_runner


def test_parse_benchmark_name_version():
    assert harbor_runner.parse_benchmark("terminal-bench-sample@2.0") == {
        "name": "terminal-bench-sample",
        "version": "2.0",
    }


def test_parse_benchmark_bare_name():
    assert harbor_runner.parse_benchmark("hello-world") == {"name": "hello-world"}


def test_parse_benchmark_local_path(tmp_path):
    out = harbor_runner.parse_benchmark(str(tmp_path))
    assert out == {"path": str(tmp_path.resolve())}


def test_primary_reward_variants():
    assert harbor_runner._primary_reward({"reward": 1.0}) == 1.0
    assert harbor_runner._primary_reward({"a": 0.5, "b": 1.5}) == 1.0
    assert harbor_runner._primary_reward({}) is None
    assert harbor_runner._primary_reward({"note": "x"}) is None


def _make_trial(job_dir, name, reward, with_traj=True):
    trial = job_dir / name
    (trial / "agent" / "sia_out").mkdir(parents=True, exist_ok=True)
    result = {
        "task_name": name,
        "verifier_result": {"rewards": {"reward": reward}} if reward is not None else {"rewards": {}},
        "exception_info": None,
    }
    (trial / "result.json").write_text(json.dumps(result), encoding="utf-8")
    if with_traj:
        (trial / "agent" / "sia_out" / "agent_execution.json").write_text(
            json.dumps({"steps": [], "usage": {}}), encoding="utf-8"
        )


def test_collect_results_aggregates_and_copies_trajectories(tmp_path):
    job_dir = tmp_path / "sia"
    job_dir.mkdir()
    (job_dir / "result.json").write_text("{}", encoding="utf-8")
    _make_trial(job_dir, "task_a", 1.0)
    _make_trial(job_dir, "task_b", 0.0)

    gen_dir = tmp_path / "gen_1"
    gen_dir.mkdir()

    results = harbor_runner._collect_results(job_dir, gen_dir, "bench@1.0")

    assert results["n_tasks"] == 2
    assert results["n_solved"] == 1
    assert results["score"] == 0.5
    assert results["benchmark"] == "bench@1.0"

    written = json.loads((gen_dir / "results.json").read_text())
    assert written["score"] == 0.5
    assert {t["task"] for t in written["per_task"]} == {"task_a", "task_b"}

    traj_files = sorted((gen_dir / "agent_execution").glob("execution_q*.json"))
    assert len(traj_files) == 2


def test_harbor_prompt_injections_exist_and_define_contract():
    from sia import prompts

    assert "--working_dir" in prompts.HARBOR_META_PROMPT
    assert "--instruction_file" in prompts.HARBOR_META_PROMPT
    assert "--log_dir" in prompts.HARBOR_META_PROMPT
    assert "submission" in prompts.HARBOR_META_PROMPT.lower()
    assert "HARBOR MODE" in prompts.HARBOR_FEEDBACK_PROMPT


def test_build_meta_prompt_default_has_no_harbor_block():
    """Default (non-harbor) path must stay byte-identical for golden snapshots."""
    from sia.prompts import build_meta_prompt
    from sia.run_setup import TaskFiles

    tf = TaskFiles("desc", "ref", {}, "task")
    assert "HARBOR MODE" not in build_meta_prompt(tf, "m", "/wd")
    assert "HARBOR MODE" in build_meta_prompt(tf, "m", "/wd", harbor=True)
