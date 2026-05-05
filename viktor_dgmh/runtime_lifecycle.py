from __future__ import annotations

import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import RestartRequest
from .paths import daemon_state_path, restart_requested_path, runtime_dir
from .serialization import read_json, write_json


def request_restart(root: Path, reason: str, *, source: str = "cli", requested_by: str | None = None) -> RestartRequest:
    request = RestartRequest(
        request_id=f"restart_{uuid.uuid4().hex}",
        reason=reason,
        source=source,
        requested_by=requested_by,
    )
    path = restart_requested_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, request.model_dump())
    return request


def load_restart_request(root: Path) -> RestartRequest | None:
    data = read_json(restart_requested_path(root), None)
    return RestartRequest.model_validate(data) if data else None


def write_restart_request(root: Path, request: RestartRequest) -> None:
    path = restart_requested_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, request.model_dump())


def runtime_status(root: Path) -> dict:
    request = load_restart_request(root)
    daemon_state = read_json(daemon_state_path(root), {}) or {}
    return {
        "restart_request": request.model_dump() if request else None,
        "daemon_state": daemon_state,
    }


@dataclass
class SlackSupervisor:
    root: Path
    command: list[str]
    poll_interval_seconds: float = 1.0
    terminate_timeout_seconds: float = 10.0
    max_restarts: int = 3
    restart_window_seconds: int = 600
    process: subprocess.Popen | None = None
    restart_history: list[datetime] = field(default_factory=list)

    @classmethod
    def for_slack(cls, root: Path, *, use_fake: bool = False) -> "SlackSupervisor":
        command = [sys.executable, "-m", "viktor_dgmh", "slack", "serve"]
        if use_fake:
            command.append("--fake")
        return cls(root=root, command=command)

    def serve_forever(self) -> None:
        runtime_dir(self.root).mkdir(parents=True, exist_ok=True)
        self.start_child()
        try:
            while True:
                self.tick()
                time.sleep(self.poll_interval_seconds)
        finally:
            self.stop_child()

    def tick(self) -> None:
        if self.process is None or self.process.poll() is not None:
            self.start_child()
        request = load_restart_request(self.root)
        if request is None or request.status != "pending":
            return
        if not self.restart_allowed():
            request.status = "blocked"
            write_restart_request(self.root, request)
            self.write_state(status="blocked", last_request_id=request.request_id, blocked_reason="restart loop guard")
            return
        self.restart_child()
        request.status = "handled"
        write_restart_request(self.root, request)
        self.write_state(status="running", last_request_id=request.request_id, last_restart_at=_now())

    def start_child(self) -> None:
        self.process = subprocess.Popen(self.command, cwd=str(self.root))
        self.write_state(status="running", child_pid=self.process.pid, command=self.command)

    def stop_child(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=self.terminate_timeout_seconds)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=self.terminate_timeout_seconds)

    def restart_child(self) -> None:
        self.stop_child()
        self.restart_history.append(datetime.now(timezone.utc))
        self.start_child()

    def restart_allowed(self) -> bool:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.restart_window_seconds)
        self.restart_history = [stamp for stamp in self.restart_history if stamp >= cutoff]
        return len(self.restart_history) < self.max_restarts

    def write_state(self, **state) -> None:
        path = daemon_state_path(self.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = read_json(path, {}) or {}
        existing.update({"updated_at": _now(), **state})
        write_json(path, existing)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
