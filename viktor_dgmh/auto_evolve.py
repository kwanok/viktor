from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

from .archive import get_active_agent_id
from .llm import ChatProvider
from .memory import load_imitation_cases, load_recent_chat_events
from .models import Config, RunState
from .paths import auto_evolve_state_path
from .reflection import reflect_on_recent_conversation
from .runner import run_evolution
from .serialization import read_json, write_json


def auto_evolve_status(root: Path, config: Config) -> tuple[bool, str]:
    if not config.auto_evolve_on_conversation:
        return False, "auto evolution is disabled"
    events = load_recent_chat_events(root, limit=config.auto_evolve_min_chat_events)
    if len(events) < config.auto_evolve_min_chat_events:
        return False, f"need {config.auto_evolve_min_chat_events} chat events, have {len(events)}"

    state = read_json(auto_evolve_state_path(root), {}) or {}
    if state.get("running"):
        started_at = state.get("started_at")
        if started_at:
            try:
                elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(started_at)).total_seconds()
            except ValueError:
                elapsed = 0
            if elapsed >= config.auto_evolve_stale_running_seconds:
                return True, "recovered stale auto evolution state"
        return False, "auto evolution is already running"
    last_run_at = state.get("last_run_at")
    if last_run_at:
        try:
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last_run_at)).total_seconds()
        except ValueError:
            elapsed = config.auto_evolve_cooldown_seconds
        if elapsed < config.auto_evolve_cooldown_seconds:
            remaining = int(config.auto_evolve_cooldown_seconds - elapsed)
            return False, f"cooldown active for {remaining}s"
    return True, "ready"


def run_auto_evolve_once(
    root: Path,
    config: Config,
    provider: ChatProvider,
    *,
    use_fake: bool = False,
    reason: str = "conversation_reflection",
) -> RunState | None:
    ready, status = auto_evolve_status(root, config)
    if not ready:
        _write_state(root, {"running": False, "last_blocked_reason": status, "last_checked_at": _now()})
        return None

    _write_state(root, {"running": True, "reason": reason, "started_at": _now(), "active_before": get_active_agent_id(root)})
    try:
        signals, cases, gaps = reflect_on_recent_conversation(root, provider, use_fake=use_fake)
        total_cases = len(load_imitation_cases(root))
        if total_cases < config.auto_evolve_min_cases:
            _write_state(
                root,
                {
                    "running": False,
                    "last_run_at": _now(),
                    "last_blocked_reason": f"reflection produced {len(cases)} new cases; total {total_cases}",
                    "last_reflection_signals": len(signals),
                    "last_capability_gaps": len(gaps),
                },
            )
            return None

        state = run_evolution(
            root,
            generations=config.auto_evolve_generations,
            children=config.auto_evolve_children,
            provider=provider,
            use_fake=use_fake,
            config=config,
        )
    except Exception as exc:
        _write_state(
            root,
            {
                "running": False,
                "last_run_at": _now(),
                "last_error": f"{type(exc).__name__}: {exc}",
                "active_after": get_active_agent_id(root),
            },
        )
        raise

    _write_state(
        root,
        {
            "running": False,
            "last_run_at": _now(),
            "last_run_id": state.run_id,
            "last_reason": reason,
            "last_capability_gaps": len(gaps),
            "active_after": get_active_agent_id(root),
            "promoted": any(event["event"] == "maybe_promote" and event["payload"].get("promoted") for event in state.events),
        },
    )
    return state


def schedule_auto_evolve_after_conversation(
    root: Path,
    config: Config,
    provider: ChatProvider,
    *,
    use_fake: bool = False,
    reason: str = "conversation",
    logger=None,
    synchronous: bool = False,
) -> bool:
    ready, status = auto_evolve_status(root, config)
    if not ready:
        if logger:
            logger.info("Auto evolution skipped: %s", status)
        return False

    def task() -> None:
        try:
            run_auto_evolve_once(root, config, provider, use_fake=use_fake, reason=reason)
        except Exception:
            if logger:
                logger.exception("Auto evolution failed")

    if synchronous:
        task()
    else:
        threading.Thread(target=task, name="viktor-auto-evolve", daemon=True).start()
    return True


def _write_state(root: Path, state: dict) -> None:
    path = auto_evolve_state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_json(path, {}) or {}
    existing.update(state)
    write_json(path, existing)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
