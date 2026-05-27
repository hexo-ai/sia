#!/usr/bin/env python3
"""
Reference target agent for the tau2-bench task.

SIA improves this file across generations. The improvable parts are:
  - AGENT_INSTRUCTION   : the system prompt injected into SIAAgent
  - SIAAgent            : subclass of LLMAgent — can override generate_next_message
                          for retry logic, multi-step planning, etc.

tau2-bench's orchestrator still handles the conversation loop, user simulator,
tool execution, and evaluation. We only own the agent's "brain".

Usage:
    python reference_target_agent.py \
        --dataset_dir /path/to/data/public \
        --working_dir  /path/to/working \
        --shared_dir   /path/to/tasks/_shared \
        --model        gemini/gemini-2.5-flash
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── What SIA iterates on ───────────────────────────────────────────────────────

AGENT_INSTRUCTION = """
You are a professional airline customer service agent.

Guidelines:
- Always query the database before making changes (never assume booking details).
- Confirm key information with the customer before irreversible actions (cancellations, refunds).
- Follow the policy strictly — do not offer exceptions not listed.
- When the customer's request is fully resolved, send a short closing message.
- Be concise: one tool call or one message per turn, not both.
""".strip()


# ── tau2-bench bootstrap ───────────────────────────────────────────────────────

def _bootstrap_tau2(dataset_dir: str) -> None:
    episodes_dir = os.path.join(dataset_dir, "episodes")
    data_file = os.path.join(episodes_dir, "tau2_data_dir.txt")
    src_file  = os.path.join(episodes_dir, "tau2_src_dir.txt")
    if os.path.exists(data_file):
        os.environ.setdefault("TAU2_DATA_DIR", Path(data_file).read_text().strip())
    if os.path.exists(src_file):
        src = Path(src_file).read_text().strip()
        if src not in sys.path:
            sys.path.insert(0, src)


def load_task_ids(dataset_dir: str) -> list[str]:
    path = os.path.join(dataset_dir, "episodes", "task_ids.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"task_ids.json not found at {path}. "
            "Run: python tasks/tau2-bench/download_data.py"
        )
    with open(path) as f:
        return json.load(f)


# ── Custom agent — registered in tau2's registry ───────────────────────────────

def _register_sia_agent() -> None:
    """Register SIAAgent under the name 'sia_agent' in tau2's registry."""
    from tau2.agent.llm_agent import LLMAgent
    from tau2.registry import registry

    class SIAAgent(LLMAgent):
        @property
        def system_prompt(self) -> str:
            return (
                f"<instructions>\n{AGENT_INSTRUCTION}\n</instructions>\n"
                f"<policy>\n{self.domain_policy}\n</policy>"
            )

    registry.register_agent_factory(
        lambda tools, domain_policy, **kwargs: SIAAgent(
            tools=tools,
            domain_policy=domain_policy,
            llm=kwargs.get("llm"),
            llm_args=kwargs.get("llm_args"),
        ),
        "sia_agent",
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--working_dir",  required=True)
    parser.add_argument("--shared_dir",   required=True)
    parser.add_argument("--model",        default="gemini/gemini-2.5-flash",
                        help="Agent LLM (default: gemini/gemini-2.5-flash)")
    parser.add_argument("--user_model",   default="openai/gpt-4o-mini",
                        help="User simulator LLM (default: openai/gpt-4o-mini)")
    parser.add_argument("--max_turns",              type=int,   default=20)
    parser.add_argument("--target_agent_timeout",   type=int,   default=1800)
    parser.add_argument("--task_model_temperature", type=float, default=0.0)
    args = parser.parse_args()

    dataset_dir      = os.path.abspath(args.dataset_dir)
    working_dir      = os.path.abspath(args.working_dir)
    os.makedirs(working_dir, exist_ok=True)

    predictions_path = os.path.join(working_dir, "predictions.json")
    evaluate_path    = os.path.join(dataset_dir, "evaluate.py")
    results_path     = os.path.join(working_dir, "results.json")

    _bootstrap_tau2(dataset_dir)

    try:
        from tau2.run import get_tasks, run_single_task
        from tau2.data_model.simulation import TextRunConfig
    except ImportError as e:
        logger.error(f"Could not import tau2: {e}\nInstall: pip install -e /tmp/tau2-bench-src")
        sys.exit(1)

    _register_sia_agent()

    task_ids = load_task_ids(dataset_dir)
    tasks    = get_tasks("airline", task_ids=task_ids)
    logger.info(f"Loaded {len(tasks)} tasks. Agent: {args.model}  User: {args.user_model}")

    config = TextRunConfig(
        domain="airline",
        agent="sia_agent",
        llm=args.model,
        llm_user=args.user_model,
        llm_args={"temperature": args.task_model_temperature},
    )

    episode_results = []
    start_time = time.time()

    for i, task in enumerate(tasks):
        elapsed   = time.time() - start_time
        remaining = args.target_agent_timeout - elapsed
        if remaining < 60:
            logger.warning(f"Timeout — stopping after {i} episodes.")
            break

        logger.info(f"Episode {i + 1}/{len(tasks)}  task_id={task.id}  ({elapsed:.0f}s elapsed)")

        try:
            sim_run = run_single_task(config, task, seed=42)
            reward  = float(sim_run.reward_info.reward) if sim_run.reward_info else 0.0
            n_turns = len([m for m in sim_run.messages if m.role == "assistant"])
            error   = None
        except Exception as e:
            logger.error(f"  Episode {task.id} crashed: {e}")
            reward, n_turns, error = 0.0, 0, str(e)

        episode_results.append({
            "task_id":   task.id,
            "reward":    reward,
            "num_turns": n_turns,
            "error":     error,
        })
        logger.info(f"  reward={reward}  turns={n_turns}")

        with open(predictions_path, "w") as f:
            json.dump({"episodes": episode_results}, f, indent=2)

    logger.info("All episodes done — calling evaluate.py...")
    os.system(f"{sys.executable} {evaluate_path} {predictions_path}")

    if not os.path.exists(results_path):
        logger.error("evaluate.py did not produce results.json")
        sys.exit(1)

    with open(results_path) as f:
        r = json.load(f)
    logger.info(f"Score: {r.get('score', 0):.4f}  ({r.get('n_passed', '?')}/{r.get('n_total', '?')} passed)")


if __name__ == "__main__":
    main()
