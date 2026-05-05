from __future__ import annotations

from pathlib import Path


def workspace_root(cwd: Path | None = None) -> Path:
    return (cwd or Path.cwd()).resolve()


def archive_dir(root: Path) -> Path:
    return root / "archive"


def benchmark_dir(root: Path) -> Path:
    return root / "benchmark"


def runs_dir(root: Path) -> Path:
    return root / "runs"


def config_path(root: Path) -> Path:
    return root / "config.yaml"


def active_path(root: Path) -> Path:
    return archive_dir(root) / "ACTIVE"

