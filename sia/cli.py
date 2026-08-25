"""Command-line argument parsing and resolution for the SIA orchestrator.

Extracted from orchestrator.main() so the parser and the arg→params resolution
can be tested independently. main() remains the entry point and dispatches on the
sub-command (``run`` / ``web``).

Sub-commands:
    sia run [flags]   Run the orchestrator (agent evolution). This is the default
                      when no sub-command is given, so ``sia --task gpqa`` still
                      works. A live dashboard is started in a background thread
                      unless ``--no-web`` is passed.
    sia web [flags]   Serve the runs visualizer over HTTP.

Agent configuration is selected via JSON *profiles* (see sia/profiles.py): the
meta/feedback agent via ``--meta-agent-profile`` and the target agent via
``--target-agent-profile``. Each value is a bundled/user profile name or a path to a
``.json`` file.
"""

from __future__ import annotations

import argparse

from sia.config import Config
from sia.layout import BUNDLED_TASKS, Names
from sia.logging_setup import get_logger

logger = get_logger(__name__)

_SUBCOMMANDS = ("run", "web")


def _add_run_args(parser: argparse.ArgumentParser, env_config: Config) -> None:
    """Attach orchestrator (agent-evolution) arguments to ``parser``."""
    parser.add_argument(
        "--max_gen",
        type=int,
        default=env_config.DEFAULT_MAX_GENERATIONS,
        help="Maximum number of generations to run (default: 3)",
    )
    parser.add_argument("--run_id", type=int, default=1, help="Run ID for this experiment (default: 1)")
    task_group = parser.add_mutually_exclusive_group(required=False)
    task_group.add_argument(
        "--task",
        type=str,
        choices=BUNDLED_TASKS,
        help=f"Name of a bundled task shipped with sia-agent ({', '.join(BUNDLED_TASKS)})",
    )
    task_group.add_argument(
        "--task_dir",
        type=str,
        help="Path to an external task directory (e.g., ./tasks/my-task)",
    )
    parser.add_argument(
        "--harbor_dataset",
        type=str,
        default=None,
        help="Harbor benchmark to run on, e.g. 'terminal-bench-sample@2.0' (enables Harbor mode)",
    )
    parser.add_argument(
        "--harbor_task_dir",
        type=str,
        default=None,
        help="Path to a pre-downloaded Harbor benchmark directory (enables Harbor mode)",
    )
    parser.add_argument(
        "--harbor_include_task",
        action="append",
        default=None,
        metavar="TASK_NAME",
        help="Restrict the Harbor run to these task name(s); repeatable",
    )
    parser.add_argument(
        "--harbor_working_dir",
        type=str,
        default="/app",
        help="Container working directory the agent operates in (default: /app)",
    )
    parser.add_argument(
        "--meta-agent-profile",
        dest="meta_agent_profile",
        type=str,
        default=env_config.DEFAULT_META_AGENT_PROFILE,
        help=(
            "Agent profile for the meta/feedback agent: a bundled/user profile name or a path "
            f"to a .json file (default: {env_config.DEFAULT_META_AGENT_PROFILE}). A profile bundles "
            "agent_impl + model + provider."
        ),
    )
    parser.add_argument(
        "--target-agent-profile",
        dest="target_agent_profile",
        type=str,
        default=env_config.DEFAULT_TARGET_AGENT_PROFILE,
        help=(
            "Agent profile for the target agent: a bundled/user profile name or a path to a "
            f".json file (default: {env_config.DEFAULT_TARGET_AGENT_PROFILE}). The model + provider "
            "the generated target_agent.py will call, plus its agent_reference (seed code)."
        ),
    )
    parser.add_argument(
        "--sandbox",
        type=str,
        default=env_config.SANDBOX_MODE,
        choices=["none", "docker"],
        help="Sandbox mode for target agent execution: none (default) or docker (requires Docker)",
    )
    parser.add_argument(
        "--focus",
        type=str,
        default="harness",
        choices=["harness", "weights"],
        help="Focus of improvement: 'harness' (default) for code/prompt changes or 'weights' for RL-based weight tuning",
    )
    parser.add_argument(
        "--training_sandbox",
        dest="training_sandbox",
        type=str,
        default="modal",
        choices=["modal", "sandboxfusion"],
        help="Sandbox for train.py code execution in weights mode (default: modal). Use 'sandboxfusion' for SandboxFusion service.",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO, or the $SIA_LOG_LEVEL env var).",
    )
    # Live dashboard controls (the web server auto-started during a run).
    parser.add_argument(
        "--no-web",
        dest="no_web",
        action="store_true",
        help="Do not start the live visualizer dashboard during the run.",
    )
    parser.add_argument(
        "--web-port",
        dest="web_port",
        type=int,
        default=8000,
        help="Port for the live dashboard started during the run (default: 8000).",
    )
    parser.add_argument(
        "--web-host",
        dest="web_host",
        type=str,
        default="127.0.0.1",
        help="Host for the live dashboard (default: 127.0.0.1).",
    )


def _add_web_args(parser: argparse.ArgumentParser, env_config: Config) -> None:
    """Attach visualizer (``sia web``) arguments to ``parser``."""
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument(
        "--runs-dir",
        dest="runs_dir",
        type=str,
        default=Names.RUNS_ROOT,
        help=f"Directory of runs to visualize (default: {Names.RUNS_ROOT}).",
    )
    parser.add_argument(
        "--no-browser",
        dest="no_browser",
        action="store_true",
        help="Do not open a browser window automatically.",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO, or the $SIA_LOG_LEVEL env var).",
    )


def build_parser(env_config: Config) -> argparse.ArgumentParser:
    """Build the top-level ``sia`` parser with ``run`` / ``web`` sub-commands."""
    parser = argparse.ArgumentParser(prog="sia", description="SIA: Self-Improving AI framework")
    sub = parser.add_subparsers(dest="command", metavar="{run,web}")

    run_parser = sub.add_parser("run", help="Run the orchestrator (agent evolution).")
    _add_run_args(run_parser, env_config)

    web_parser = sub.add_parser("web", help="Serve the runs visualizer over HTTP.")
    _add_web_args(web_parser, env_config)

    return parser


def parse_args(env_config: Config, argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments, defaulting to the ``run`` sub-command.

    For backward compatibility ``sia --task gpqa`` (no sub-command) is treated as
    ``sia run --task gpqa``; ``sia web ...`` opts into the visualizer.
    """
    import sys

    raw = sys.argv[1:] if argv is None else argv
    # Insert the default sub-command unless the user asked for one (or for help).
    if not raw or (raw[0] not in _SUBCOMMANDS and raw[0] not in ("-h", "--help")):
        raw = ["run", *raw]

    args = build_parser(env_config).parse_args(raw)
    if args.command is None:
        args.command = "run"
    return args
