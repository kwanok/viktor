from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

from .models import Config, ShellCommandRecord
from .paths import shell_commands_path
from .serialization import append_jsonl


SHELL_PREFIXES = ("!sh ", "!bash ")


def parse_shell_command(text: str) -> str | None:
    stripped = text.strip()
    for prefix in SHELL_PREFIXES:
        if stripped.startswith(prefix):
            command = stripped[len(prefix) :].strip()
            return command or None
    return None


def shell_enabled(config: Config) -> bool:
    return _env_bool("SLACK_ENABLE_SHELL", config.slack_shell_enabled)


def shell_user_allowed(user: str | None) -> bool:
    if _env_bool("SLACK_SHELL_ALLOW_ANY", False):
        return True
    allowed = {
        item.strip()
        for item in os.environ.get("SLACK_SHELL_ALLOWED_USERS", "").split(",")
        if item.strip()
    }
    return bool(user and user in allowed)


def run_shell_command(
    root: Path,
    command: str,
    *,
    config: Config,
    user: str | None = None,
    channel: str | None = None,
    slack_ts: str | None = None,
) -> ShellCommandRecord:
    timed_out = False
    try:
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=config.slack_shell_timeout_seconds,
            check=False,
        )
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stderr = (stderr + "\n" if stderr else "") + f"Timed out after {config.slack_shell_timeout_seconds}s."

    max_chars = config.slack_shell_max_output_chars
    record = ShellCommandRecord(
        command_id=f"shell_{uuid.uuid4().hex}",
        user=user,
        channel=channel,
        slack_ts=slack_ts,
        command=command,
        cwd=str(root),
        exit_code=exit_code,
        stdout=_truncate(stdout, max_chars),
        stderr=_truncate(stderr, max_chars),
        timed_out=timed_out,
    )
    append_jsonl(shell_commands_path(root), record.model_dump())
    return record


def format_shell_result(record: ShellCommandRecord) -> str:
    output = record.stdout or ""
    error = record.stderr or ""
    body = output if output else "(no stdout)"
    if error:
        body += f"\n\nstderr:\n{error}"
    body = body.replace("```", "` ` `")
    return f"exit={record.exit_code} timed_out={str(record.timed_out).lower()}\n```text\n{body}\n```"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 40] + "\n... [truncated]"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}
