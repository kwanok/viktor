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


def memory_dir(root: Path) -> Path:
    return root / "memory"


def chat_sessions_dir(root: Path) -> Path:
    return memory_dir(root) / "chat_sessions"


def preferences_path(root: Path) -> Path:
    return memory_dir(root) / "preferences.jsonl"


def imitation_cases_path(root: Path) -> Path:
    return memory_dir(root) / "imitation_cases.jsonl"


def slack_message_map_path(root: Path) -> Path:
    return memory_dir(root) / "slack_message_map.jsonl"


def router_dir(root: Path) -> Path:
    return root / "router"


def router_active_path(root: Path) -> Path:
    return router_dir(root) / "active.yaml"


def router_archive_dir(root: Path) -> Path:
    return router_dir(root) / "archive"


def router_observations_path(root: Path) -> Path:
    return memory_dir(root) / "router_observations.jsonl"


def router_labels_path(root: Path) -> Path:
    return memory_dir(root) / "router_labels.jsonl"


def shell_commands_path(root: Path) -> Path:
    return memory_dir(root) / "shell_commands.jsonl"


def config_path(root: Path) -> Path:
    return root / "config.yaml"


def active_path(root: Path) -> Path:
    return archive_dir(root) / "ACTIVE"
