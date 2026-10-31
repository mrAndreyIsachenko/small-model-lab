"""Hardware, software, and git provenance capture for reproducible runs."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return proc.stdout.strip()
    except Exception:
        return None


def compute_file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_environment(repo_root: Path | None = None) -> dict[str, Any]:
    """Capture host OS, Python, PyTorch, accelerator, git, and lockfile state."""
    root = repo_root if repo_root is not None else Path.cwd()
    git_sha = _run_git(["rev-parse", "HEAD"], cwd=root) or "uncommitted"
    git_status = _run_git(["status", "--porcelain"], cwd=root)
    git_dirty = bool(git_status)

    accelerator = "cpu"
    device_name = platform.processor() or platform.machine()
    vram_gb: float | None = None
    cuda_version: str | None = torch.version.cuda

    if torch.cuda.is_available():
        accelerator = "cuda"
        device_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / (1024**3), 2)
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        device_name = f"Apple Silicon MPS ({platform.machine()})"

    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "accelerator": accelerator,
        "device_name": device_name,
        "vram_gb": vram_gb,
        "cuda_version": cuda_version,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "uv_lock_sha256": compute_file_sha256(root / "uv.lock"),
    }
