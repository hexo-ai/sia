"""
tools.py — sandboxed tool implementations for target agents.

Usage in your target_agent.py:
    import sys
    sys.path.insert(0, shared_dir)   # shared_dir passed via --shared_dir
    from tools import bash, read_file, write_file

Each function enforces path restrictions:
  - bash:       only absolute paths inside working_dir or dataset_dir (+ system paths)
  - read_file:  only inside working_dir or dataset_dir
  - write_file: only inside working_dir
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
from pathlib import Path

SYSTEM_PATHS = ("/usr", "/bin", "/lib", "/etc", "/tmp", "/dev", "/proc", "/sys")


def _bash_paths_allowed(command: str, working_dir: str, dataset_dir: str) -> bool:
    allowed = (Path(working_dir).resolve(), Path(dataset_dir).resolve())
    for m in re.finditer(r"(?<!\w)(\/[^\s\"'|;&><,()]+)", command):
        p = Path(m.group(1)).resolve()
        if any(p.is_relative_to(d) for d in allowed):
            continue
        if any(str(p).startswith(s) for s in SYSTEM_PATHS):
            continue
        return False
    return True


def bash(command: str, *, working_dir: str, dataset_dir: str) -> str:
    if not _bash_paths_allowed(command, working_dir, dataset_dir):
        return "[ERROR] bash command references a path outside working_dir or dataset_dir"
    try:
        proc = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True, cwd=working_dir,
        )
        try:
            stdout, stderr = proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                proc.kill()
            proc.wait()
            return "[ERROR] timed out"
        out = stdout + (f"\n[stderr]\n{stderr}" if stderr else "")
        return out.strip() or "(no output)"
    except Exception as e:
        return f"[ERROR] {e}"


def _resolve_path(path: str, base_dir: str) -> Path:
    """Resolve path relative to base_dir if not absolute."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(base_dir) / p
    return p.resolve()


def read_file(path: str, *, working_dir: str, dataset_dir: str) -> str:
    try:
        p = _resolve_path(path, working_dir)
        allowed = (Path(working_dir).resolve(), Path(dataset_dir).resolve())
        if not any(p.is_relative_to(d) for d in allowed):
            return "[ERROR] read_file path must be inside working_dir or dataset_dir"
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"[ERROR] {e}"


def write_file(path: str, content: str, *, working_dir: str) -> str:
    try:
        p = _resolve_path(path, working_dir)
        if not p.is_relative_to(Path(working_dir).resolve()):
            return "[ERROR] write_file path must be inside working_dir"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {p}"
    except Exception as e:
        return f"[ERROR] {e}"
