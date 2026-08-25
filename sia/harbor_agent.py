"""Harbor agent that runs a SIA-generated target agent inside a benchmark task container."""

import json
import os
import shlex
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

FORWARDED_KEY_VARS = (
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
)
AGENT_DIR = "/tmp/sia_agent"
OUTPUT_DIR = "/tmp/sia_agent/out"


async def _maybe_await(value):
    return await value if hasattr(value, "__await__") else value


class SIATargetAgent(BaseAgent):
    def __init__(
        self,
        *args,
        agent_path: str | None = None,
        working_dir: str = "/app",
        task_model: str | None = None,
        run_timeout: int = 900,
        pip_packages: str = "anthropic",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not agent_path:
            raise ValueError("SIATargetAgent requires agent_path (the generated target_agent.py)")
        self._agent_path = str(Path(agent_path).expanduser().resolve())
        self._working_dir = working_dir
        self._task_model = task_model or self.model_name
        self._run_timeout = int(run_timeout)
        self._pip_packages = pip_packages.strip()

    @staticmethod
    def name() -> str:
        return "sia-target-agent"

    def version(self) -> str:
        return "0.1.0"

    def _agent_env(self) -> dict:
        env = {k: os.environ[k] for k in FORWARDED_KEY_VARS if os.environ.get(k)}
        if self._task_model:
            env["SIA_TASK_MODEL"] = self._task_model
        return env

    async def setup(self, environment: BaseEnvironment) -> None:
        await _maybe_await(
            environment.exec(
                "command -v python3 >/dev/null 2>&1 || (apt-get update && apt-get install -y python3 python3-pip)",
                timeout_sec=600,
            )
        )
        if self._pip_packages:
            await _maybe_await(
                environment.exec(
                    f"python3 -m pip install --quiet {self._pip_packages} --break-system-packages "
                    f"|| python3 -m pip install --quiet {self._pip_packages}",
                    timeout_sec=600,
                )
            )
        await _maybe_await(environment.exec(f"mkdir -p {AGENT_DIR} {OUTPUT_DIR}"))

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        await _maybe_await(environment.upload_file(self._agent_path, f"{AGENT_DIR}/target_agent.py"))
        instruction_host = Path(self.logs_dir) / "INSTRUCTION.md"
        instruction_host.write_text(instruction, encoding="utf-8")
        await _maybe_await(environment.upload_file(str(instruction_host), f"{AGENT_DIR}/INSTRUCTION.md"))

        cmd = (
            f"cd {shlex.quote(self._working_dir)} && "
            f"python3 {AGENT_DIR}/target_agent.py "
            f"--working_dir {shlex.quote(self._working_dir)} "
            f"--instruction_file {AGENT_DIR}/INSTRUCTION.md "
            f"--log_dir {OUTPUT_DIR}"
        )
        result = await _maybe_await(environment.exec(cmd, env=self._agent_env(), timeout_sec=self._run_timeout))

        stdout = getattr(result, "stdout", "") or ""
        stderr = getattr(result, "stderr", "") or ""
        return_code = getattr(result, "return_code", None)
        (Path(self.logs_dir) / "agent_stdout.log").write_text(stdout + "\n" + stderr, encoding="utf-8")

        try:
            await _maybe_await(environment.download_dir(OUTPUT_DIR, str(Path(self.logs_dir) / "sia_out")))
        except Exception as exc:
            (Path(self.logs_dir) / "download_error.txt").write_text(str(exc), encoding="utf-8")

        self._populate_context(context, return_code)

    def _populate_context(self, context: AgentContext, return_code) -> None:
        traj = Path(self.logs_dir) / "sia_out" / "agent_execution.json"
        if not traj.is_file():
            return
        try:
            usage = json.loads(traj.read_text(encoding="utf-8")).get("usage", {})
        except Exception:
            return
        if usage.get("input_tokens"):
            context.n_input_tokens = usage["input_tokens"]
        if usage.get("output_tokens"):
            context.n_output_tokens = usage["output_tokens"]
        context.metadata = {**(context.metadata or {}), "agent_return_code": return_code}
