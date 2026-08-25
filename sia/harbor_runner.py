"""Drive a Harbor job for one SIA generation and fold the result back into SIA."""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ADAPTER_IMPORT_NAME = "sia_harbor_adapter"
ADAPTER_CLASS = "SIATargetAgent"
JOB_NAME = "sia"


@dataclass
class HarborRun:
    benchmark_path: str
    task_model: str
    working_dir: str
    include_tasks: list[str] | None


def resolve_harbor_bin() -> str:
    harbor_bin = os.getenv("SIA_HARBOR_BIN") or shutil.which("harbor")
    if not harbor_bin:
        raise SystemExit(
            "Harbor CLI not found. Install it (e.g. `uv tool install harbor`) and ensure "
            "`harbor` is on PATH, or set SIA_HARBOR_BIN to its full path."
        )
    return harbor_bin


def parse_benchmark(spec: str) -> dict:
    path = Path(spec).expanduser()
    if path.exists():
        return {"path": str(path.resolve())}
    if "@" in spec:
        name, version = spec.split("@", 1)
        return {"name": name, "version": version}
    return {"name": spec}


def prepare_harbor_benchmark(benchmark: str, dest_dir: str, n_samples: int = 3) -> tuple[str, str]:
    """Resolve a benchmark to a local task-parent dir + sample instructions.

    A local directory is used as-is; a ``name@version`` spec is downloaded (export
    layout) into ``dest_dir/benchmark/``. Returns ``(benchmark_path, sample_text)``
    where benchmark_path is a directory whose children are task folders.
    """
    p = Path(benchmark).expanduser()
    if p.exists():
        parent = p.resolve()
    else:
        out = Path(dest_dir, "benchmark")
        out.mkdir(parents=True, exist_ok=True)
        harbor_bin = resolve_harbor_bin()
        logger.info("Downloading Harbor benchmark '%s' to %s", benchmark, out)
        subprocess.run([harbor_bin, "download", benchmark, "--export", "-o", str(out)], check=True)
        instr = sorted(out.glob("*/*/instruction.md")) or sorted(out.glob("**/instruction.md"))
        if not instr:
            raise SystemExit(f"No tasks (instruction.md) found after downloading '{benchmark}'")
        parent = instr[0].parent.parent

    samples = []
    for ip in sorted(parent.glob("*/instruction.md"))[:n_samples]:
        text = ip.read_text(encoding="utf-8")
        if len(text) > 1500:
            text = text[:1500] + "\n...(truncated)"
        samples.append(f"### Task: {ip.parent.name}\n{text}")
    sample_text = "\n\n".join(samples) if samples else "(no sample instructions found)"
    return str(parent), sample_text


def _build_config(
    *,
    target_agent_path,
    task_model,
    benchmark,
    jobs_dir,
    working_dir,
    include_tasks,
    n_concurrent,
    run_timeout,
) -> dict:
    dataset = parse_benchmark(benchmark)
    if include_tasks:
        dataset["task_names"] = include_tasks
    return {
        "job_name": JOB_NAME,
        "jobs_dir": jobs_dir,
        "n_concurrent_trials": n_concurrent,
        "agents": [
            {
                "import_path": f"{ADAPTER_IMPORT_NAME}:{ADAPTER_CLASS}",
                "model_name": task_model,
                "kwargs": {
                    "agent_path": target_agent_path,
                    "task_model": task_model,
                    "working_dir": working_dir,
                    "run_timeout": run_timeout,
                },
            }
        ],
        "datasets": [dataset],
    }


def _stage_adapter(staging_dir: Path) -> None:
    src = Path(__file__).with_name("harbor_agent.py")
    shutil.copy(src, staging_dir / f"{ADAPTER_IMPORT_NAME}.py")


def _primary_reward(rewards: dict):
    if not rewards:
        return None
    if "reward" in rewards:
        return rewards["reward"]
    numeric = [v for v in rewards.values() if isinstance(v, (int, float))]
    return sum(numeric) / len(numeric) if numeric else None


def _collect_results(job_dir: Path, gen_dir: Path, benchmark: str) -> dict:
    exec_out_dir = gen_dir / "agent_execution"
    exec_out_dir.mkdir(parents=True, exist_ok=True)

    per_task = []
    trials = sorted(p for p in job_dir.glob("*/result.json") if p.parent.name != job_dir.name)
    for idx, trial_path in enumerate(trials):
        try:
            trial = json.loads(trial_path.read_text(encoding="utf-8"))
        except Exception as exc:
            per_task.append({"task": trial_path.parent.name, "error": f"unreadable result: {exc}"})
            continue
        rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
        reward = _primary_reward(rewards)
        per_task.append(
            {
                "task": trial.get("task_name", trial_path.parent.name),
                "reward": reward,
                "rewards": rewards,
                "solved": bool(reward and reward >= 1.0),
                "exception": trial.get("exception_info"),
            }
        )
        traj = trial_path.parent / "agent" / "sia_out" / "agent_execution.json"
        if traj.is_file():
            shutil.copy(traj, exec_out_dir / f"execution_q{idx}.json")

    rewards_vals = [t["reward"] for t in per_task if isinstance(t.get("reward"), (int, float))]
    mean_reward = sum(rewards_vals) / len(rewards_vals) if rewards_vals else 0.0
    results = {
        "benchmark": benchmark,
        "score": mean_reward,
        "mean_reward": mean_reward,
        "n_tasks": len(per_task),
        "n_solved": sum(1 for t in per_task if t.get("solved")),
        "n_errors": sum(1 for t in per_task if t.get("exception")),
        "per_task": per_task,
    }
    (gen_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def run_generation_on_harbor(
    target_agent_path: str,
    gen_dir: str,
    task_model: str,
    benchmark: str,
    *,
    working_dir: str = "/app",
    include_tasks: list[str] | None = None,
    n_concurrent: int = 2,
    run_timeout: int = 900,
) -> dict:
    harbor_bin = resolve_harbor_bin()
    gen_path = Path(gen_dir).resolve()
    jobs_dir = gen_path / "harbor_jobs"
    target_agent_path = str(Path(target_agent_path).resolve())

    with tempfile.TemporaryDirectory(prefix="sia_harbor_") as staging:
        staging_dir = Path(staging)
        _stage_adapter(staging_dir)
        config = _build_config(
            target_agent_path=target_agent_path,
            task_model=task_model,
            benchmark=benchmark,
            jobs_dir=str(jobs_dir),
            working_dir=working_dir,
            include_tasks=include_tasks,
            n_concurrent=n_concurrent,
            run_timeout=run_timeout,
        )
        config_path = staging_dir / "job.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(staging_dir), env.get("PYTHONPATH", "")]))

        logger.info("Launching Harbor job on benchmark '%s' (model=%s)", benchmark, task_model)
        proc = subprocess.run([harbor_bin, "run", "-c", str(config_path)], env=env, text=True, capture_output=True)
        (gen_path / "harbor_run.log").write_text(
            (proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or ""), encoding="utf-8"
        )
        if proc.returncode != 0:
            logger.error("Harbor run exited with code %s (see harbor_run.log)", proc.returncode)

    job_dir = jobs_dir / JOB_NAME
    if not (job_dir / "result.json").is_file():
        results = {
            "benchmark": benchmark,
            "score": 0.0,
            "mean_reward": 0.0,
            "n_tasks": 0,
            "n_solved": 0,
            "n_errors": 0,
            "per_task": [],
            "error": f"Harbor job did not produce result.json (exit {proc.returncode})",
        }
        (gen_path / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results

    results = _collect_results(job_dir, gen_path, benchmark)
    logger.info(
        "Harbor job complete: score=%.3f solved=%d/%d errors=%d",
        results["score"],
        results["n_solved"],
        results["n_tasks"],
        results["n_errors"],
    )
    return results
