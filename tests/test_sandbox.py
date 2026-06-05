"""Tests for Docker sandbox execution in orchestrator."""

from unittest.mock import MagicMock, patch

from sia.config import Config
from sia.orchestrator import _run_target_agent, _run_target_agent_sandboxed


@patch("sia.orchestrator.subprocess.Popen")
def test_docker_command_has_network_none(mock_popen):
    """Sandboxed run must include --network none."""
    mock_process = MagicMock()
    mock_process.stdout = iter([])
    mock_process.wait.return_value = 0
    mock_popen.return_value = mock_process

    _run_target_agent_sandboxed(
        python_exec="/venv/bin/python",
        target_agent_path="/work/target_agent.py",
        dataset_dir="/data",
        working_dir="/work",
        stdout_log_file="/dev/null",
        config=Config(),
    )

    cmd = mock_popen.call_args[0][0]
    assert "--network" in cmd
    idx = cmd.index("--network")
    assert cmd[idx + 1] == "none"


@patch("sia.orchestrator.subprocess.Popen")
def test_docker_dataset_mounted_readonly(mock_popen):
    """Dataset mount must have :ro suffix."""
    mock_process = MagicMock()
    mock_process.stdout = iter([])
    mock_process.wait.return_value = 0
    mock_popen.return_value = mock_process

    _run_target_agent_sandboxed(
        python_exec="/venv/bin/python",
        target_agent_path="/work/target_agent.py",
        dataset_dir="/data",
        working_dir="/work",
        stdout_log_file="/dev/null",
        config=Config(),
    )

    cmd = mock_popen.call_args[0][0]
    # Find the -v flag for dataset
    vol_idx = cmd.index("-v")
    dataset_vol = cmd[vol_idx + 1]
    assert ":/data:ro" in dataset_vol


@patch("sia.orchestrator.subprocess.Popen")
def test_docker_working_dir_mounted_readwrite(mock_popen):
    """Working dir mount must have :rw suffix."""
    mock_process = MagicMock()
    mock_process.stdout = iter([])
    mock_process.wait.return_value = 0
    mock_popen.return_value = mock_process

    _run_target_agent_sandboxed(
        python_exec="/venv/bin/python",
        target_agent_path="/work/target_agent.py",
        dataset_dir="/data",
        working_dir="/work",
        stdout_log_file="/dev/null",
        config=Config(),
    )

    cmd = mock_popen.call_args[0][0]
    # Find all -v flags
    vol_indices = [i for i, x in enumerate(cmd) if x == "-v"]
    work_vol = cmd[vol_indices[1] + 1]
    assert ":/work:rw" in work_vol


@patch("sia.orchestrator.subprocess.Popen")
def test_docker_image_and_resource_limits(mock_popen):
    """Docker command uses image and resource limits from Config."""
    mock_process = MagicMock()
    mock_process.stdout = iter([])
    mock_process.wait.return_value = 0
    mock_popen.return_value = mock_process

    cfg = Config()
    _run_target_agent_sandboxed(
        python_exec="/venv/bin/python",
        target_agent_path="/work/target_agent.py",
        dataset_dir="/data",
        working_dir="/work",
        stdout_log_file="/dev/null",
        config=cfg,
    )

    cmd = mock_popen.call_args[0][0]
    assert cfg.DOCKER_IMAGE in cmd
    assert "--memory" in cmd
    mem_idx = cmd.index("--memory")
    assert cmd[mem_idx + 1] == cfg.DOCKER_MEMORY_LIMIT
    # CPU limit in --cpus=N format
    cpu_args = [a for a in cmd if a.startswith("--cpus=")]
    assert len(cpu_args) == 1
    assert str(cfg.DOCKER_CPU_LIMIT) in cpu_args[0]


@patch("sia.orchestrator.subprocess.Popen")
def test_sandbox_none_uses_standard_popen(mock_popen):
    """sandbox='none' bypasses Docker entirely."""
    gen_dir = "/tmp/gen"
    mock_process = MagicMock()
    mock_process.stdout = iter(["line\n"])
    mock_process.wait.return_value = 0
    mock_popen.return_value = mock_process

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        log_path = f.name

    _run_target_agent(
        venv_dir="/fake/venv",
        target_agent_path=f"{gen_dir}/target_agent.py",
        abs_dataset_dir="/data",
        gen_dir=gen_dir,
        stdout_log_file=log_path,
        sandbox="none",
        env_config=Config(),
    )

    cmd = mock_popen.call_args[0][0]
    assert cmd[0] == "/fake/venv/bin/python"
    assert "docker" not in cmd[0]
