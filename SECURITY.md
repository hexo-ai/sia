# Security Model

## Agent Execution

SIA spawns AI agents that can execute code on the host system. The security implications depend on the execution mode.

### Execution Modes

| Mode | Flag | Isolation | Risk Level |
|------|------|-----------|------------|
| Direct | `--sandbox none` (default) | None | High |
| Docker | `--sandbox docker` | Container + no network | Low |

### Direct Mode (default)

In direct mode, the target agent runs as a subprocess with full access to:
- The host filesystem (within the user's permissions)
- Network resources
- Environment variables (including API keys)

This is appropriate for **trusted research environments** where:
- The task data is controlled
- The agent model is trusted
- The execution environment is isolated (VM, container, etc.)

### Docker Sandbox Mode

When `--sandbox docker` is used:
- Target agent runs in a Docker container
- Dataset directory is mounted **read-only**
- Working directory is mounted **read-write**
- Network access is **disabled** (`--network none`)
- Memory and CPU limits are enforced
- The agent cannot access the host filesystem or environment variables

Requirements: Docker must be installed and the user must have permission to run containers.

### bypassPermissions in util.py

The Claude Code SDK agent runner uses `permission_mode="bypassPermissions"`. This is required for automated agent execution -- without it, the agent would pause and wait for human approval on every file operation. This is safe when:
- Operating in a controlled workspace
- Using the Docker sandbox for untrusted tasks
- The agent model is trusted

## Reference Agent Templates

Files in `sia/tasks/_shared/` and `sia/tasks/*/reference/` are **template code** that the meta-agent reads and uses as examples. They use `subprocess.run(shell=True)` because they are not executed by the framework directly -- they are rewritten by the meta-agent into new target agents.

## Reporting

Report security vulnerabilities to security@hexo.ai.
