#!/usr/bin/env python3
"""
Download and prepare tau2-bench data for the SIA framework.

Usage:
    python download_data.py

What this does:
  1. Clones the tau2-bench repo and installs it
  2. Extracts airline domain tasks
  3. Splits into public (80 tasks) and private (20 tasks, held-out)
  4. Copies policy docs and tool definitions into data/public/ and data/private/

Run once before the first orchestrator launch.
"""

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

TASK_DIR   = Path(__file__).parent
PUBLIC_DIR = TASK_DIR / "data" / "public"
PRIV_DIR   = TASK_DIR / "data" / "private"

TAU2_REPO  = "https://github.com/sierra-research/tau2-bench"
TAU2_TMP   = Path("/tmp/tau2-bench-src")

DOMAIN      = "airline"   # primary domain for this task
N_PUBLIC    = 40          # episodes visible to the agent during development
N_PRIVATE   = 10          # held-out episodes for private scoring
RANDOM_SEED = 42


def run(cmd: str) -> None:
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def clone_and_install() -> None:
    if TAU2_TMP.exists():
        print(f"[skip] {TAU2_TMP} already exists — delete it to re-clone.")
        return
    run(f"git clone --depth 1 {TAU2_REPO} {TAU2_TMP}")
    run(f"pip install -e {TAU2_TMP} -q")


def find_tasks(src: Path, domain: str) -> list[dict]:
    """Locate the tasks JSON for the given domain inside the tau2-bench repo."""
    candidates = list(src.rglob(f"{domain}/tasks.json")) + list(src.rglob(f"{domain}/data.json"))
    if not candidates:
        # Fallback: look for any JSON with a list of task dicts
        candidates = list(src.rglob(f"*{domain}*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find tasks JSON for domain '{domain}' under {src}. "
            "Inspect the repo structure and update the path in this script."
        )
    tasks_path = candidates[0]
    print(f"Found tasks at: {tasks_path}")
    with open(tasks_path) as f:
        data = json.load(f)
    # tau2-bench stores tasks as a list directly, or under a "tasks" key
    if isinstance(data, list):
        return data
    return data.get("tasks", data.get("data", []))


def find_aux_files(src: Path, domain: str) -> dict[str, Path]:
    """Find policy doc, tools JSON, and any other aux files for the domain."""
    found = {}
    for name in ("policy.md", "policy.txt", "tools.json", "background.md"):
        hits = list(src.rglob(f"{domain}/{name}"))
        if hits:
            found[name] = hits[0]
    return found


def write_split(
    tasks: list[dict],
    dest_dir: Path,
    aux_files: dict[str, Path],
    tau2_data_dir: Path,
    tau2_src_dir: Path,
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = dest_dir / "episodes"
    episodes_dir.mkdir(exist_ok=True)

    # Task ID list — the target agent uses these to filter tau2-bench's full dataset
    task_ids = [str(t["id"]) for t in tasks]
    with open(episodes_dir / "task_ids.json", "w") as f:
        json.dump(task_ids, f, indent=2)
    print(f"  Wrote {len(task_ids)} task IDs → {episodes_dir / 'task_ids.json'}")

    # Full task data snapshot (for reference / offline inspection)
    with open(episodes_dir / "tasks.json", "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"  Wrote task snapshot → {episodes_dir / 'tasks.json'}")

    # Aux files (policy.md etc.)
    for name, src_path in aux_files.items():
        dst = episodes_dir / name
        shutil.copy2(src_path, dst)
        print(f"  Copied {name} → {dst}")

    # Store tau2 paths so target_agent.py can bootstrap without env vars
    (episodes_dir / "tau2_data_dir.txt").write_text(str(tau2_data_dir))
    (episodes_dir / "tau2_src_dir.txt").write_text(str(tau2_src_dir))
    print(f"  Stored tau2_data_dir: {tau2_data_dir}")


def main() -> None:
    print("=== tau2-bench data download ===\n")

    # 1. Clone & install
    clone_and_install()

    # 2. Load tasks
    all_tasks = find_tasks(TAU2_TMP, DOMAIN)
    print(f"\nLoaded {len(all_tasks)} '{DOMAIN}' tasks from tau2-bench.\n")

    if len(all_tasks) < N_PUBLIC + N_PRIVATE:
        print(
            f"WARNING: only {len(all_tasks)} tasks available "
            f"(need {N_PUBLIC + N_PRIVATE}). Adjusting split."
        )
        n_pub = int(len(all_tasks) * 0.8)
        n_priv = len(all_tasks) - n_pub
    else:
        n_pub, n_priv = N_PUBLIC, N_PRIVATE

    rng = random.Random(RANDOM_SEED)
    indices = list(range(len(all_tasks)))
    rng.shuffle(indices)

    public_tasks  = [all_tasks[i] for i in indices[:n_pub]]
    private_tasks = [all_tasks[i] for i in indices[n_pub: n_pub + n_priv]]

    aux          = find_aux_files(TAU2_TMP, DOMAIN)
    tau2_data_dir = TAU2_TMP / "data"
    tau2_src_dir  = TAU2_TMP / "src"

    # 3. Write public split
    print(f"Writing public split ({len(public_tasks)} tasks)...")
    write_split(public_tasks, PUBLIC_DIR, aux, tau2_data_dir, tau2_src_dir)

    # 4. Write private split
    print(f"Writing private split ({len(private_tasks)} tasks)...")
    write_split(private_tasks, PRIV_DIR, aux, tau2_data_dir, tau2_src_dir)

    print("\n=== Done ===")
    print(f"Public  : {PUBLIC_DIR / 'episodes'}")
    print(f"Private : {PRIV_DIR  / 'episodes'}")
    print("\nNext step: run ./launch.sh")


if __name__ == "__main__":
    main()
