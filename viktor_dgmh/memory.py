from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from .models import CapabilityGap, CapabilityWorkItem, ChatEvent, ImitationCase, PreferenceSignal
from .paths import (
    capability_gaps_path,
    capability_work_path,
    chat_sessions_dir,
    imitation_cases_path,
    preferences_path,
    slack_message_map_path,
)
from .serialization import append_jsonl, read_jsonl


FEEDBACK_MAP = {
    "/good": ("good_example", "positive", 0.7, "User marked the answer as good."),
    "/bad": ("bad_example", "negative", 0.7, "User marked the answer as bad."),
    "/too-long": ("too_verbose", "negative", 0.8, "Prefer shorter, more direct answers."),
    "/weak-evidence": ("weak_evidence", "negative", 0.8, "Prefer stronger evidence and clearer claim/evidence separation."),
    "/unsafe": ("unsafe", "negative", 1.0, "Prefer safer behavior and approval before risky actions."),
}


def new_session_id() -> str:
    return datetime.now(timezone.utc).strftime("chat_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]


def new_event_id(prefix: str = "evt") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def session_path(root: Path, session_id: str) -> Path:
    return chat_sessions_dir(root) / f"{session_id}.jsonl"


def append_chat_event(root: Path, event: ChatEvent) -> None:
    append_jsonl(session_path(root, event.session_id), event.model_dump())


def append_preference(root: Path, signal: PreferenceSignal) -> None:
    append_jsonl(preferences_path(root), signal.model_dump())


def append_imitation_case(root: Path, case: ImitationCase) -> None:
    append_jsonl(imitation_cases_path(root), case.model_dump())


def append_capability_gap(root: Path, gap: CapabilityGap) -> None:
    append_jsonl(capability_gaps_path(root), gap.model_dump())


def append_capability_work_item(root: Path, item: CapabilityWorkItem) -> None:
    append_jsonl(capability_work_path(root), item.model_dump())


def load_preferences(root: Path) -> list[PreferenceSignal]:
    return [PreferenceSignal.model_validate(row) for row in read_jsonl(preferences_path(root))]


def load_imitation_cases(root: Path) -> list[ImitationCase]:
    return [ImitationCase.model_validate(row) for row in read_jsonl(imitation_cases_path(root))]


def load_capability_gaps(root: Path) -> list[CapabilityGap]:
    return [CapabilityGap.model_validate(row) for row in read_jsonl(capability_gaps_path(root))]


def load_capability_work_items(root: Path) -> list[CapabilityWorkItem]:
    return [CapabilityWorkItem.model_validate(row) for row in read_jsonl(capability_work_path(root))]


def find_capability_gap(root: Path, gap_id: str) -> CapabilityGap | None:
    return next((gap for gap in load_capability_gaps(root) if gap.gap_id == gap_id), None)


def load_session_events(root: Path, session_id: str) -> list[ChatEvent]:
    return [ChatEvent.model_validate(row) for row in read_jsonl(session_path(root, session_id))]


def load_recent_chat_events(root: Path, limit: int = 40) -> list[ChatEvent]:
    base = chat_sessions_dir(root)
    if not base.exists():
        return []
    events: list[ChatEvent] = []
    for path in sorted(base.glob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True):
        for row in read_jsonl(path):
            events.append(ChatEvent.model_validate(row))
        if len(events) >= limit * 2:
            break
    events.sort(key=lambda event: event.ts)
    return events[-limit:]


def load_chat_event(root: Path, session_id: str, event_id: str) -> ChatEvent | None:
    for event in load_session_events(root, session_id):
        if event.event_id == event_id:
            return event
    return None


def append_slack_message_map(root: Path, mapping: dict) -> None:
    append_jsonl(slack_message_map_path(root), mapping)


def find_slack_message_map(root: Path, *, channel: str, ts: str) -> dict | None:
    for row in reversed(read_jsonl(slack_message_map_path(root))):
        if row.get("channel") == channel and row.get("ts") == ts:
            return row
    return None


def extract_preference_from_feedback(
    feedback: ChatEvent,
    *,
    prompt: ChatEvent | None,
    answer: ChatEvent | None,
) -> tuple[PreferenceSignal | None, ImitationCase | None]:
    text = feedback.text.strip()
    command, payload = split_feedback_command(text)
    if command is None:
        return None, None

    context = infer_context(prompt.text if prompt else "")
    target_id = answer.event_id if answer else None
    signal_id = f"sig_{uuid.uuid4().hex}"

    if command == "/remember":
        preference_text = payload or "Remember this preference."
        signal = PreferenceSignal(
            signal_id=signal_id,
            session_id=feedback.session_id,
            source_event_id=feedback.event_id,
            target_event_id=target_id,
            kind="explicit_preference",
            polarity="positive",
            strength=0.95,
            context=context,
            text=preference_text,
        )
        case = _case_from_signal(signal, prompt, answer, preference_text)
        return signal, case

    if command == "/rewrite":
        preferred_text = payload or ""
        signal = PreferenceSignal(
            signal_id=signal_id,
            session_id=feedback.session_id,
            source_event_id=feedback.event_id,
            target_event_id=target_id,
            kind="rewrite",
            polarity="positive",
            strength=1.0,
            context=context,
            text="User supplied a preferred rewrite.",
            preferred_text=preferred_text,
        )
        case = _case_from_signal(signal, prompt, answer, "Prefer the rewrite style and judgment.", preferred_text)
        return signal, case

    if command not in FEEDBACK_MAP:
        return None, None

    kind, polarity, strength, preference_text = FEEDBACK_MAP[command]
    signal = PreferenceSignal(
        signal_id=signal_id,
        session_id=feedback.session_id,
        source_event_id=feedback.event_id,
        target_event_id=target_id,
        kind=kind,
        polarity=polarity,  # type: ignore[arg-type]
        strength=strength,
        context=context,
        text=payload or preference_text,
    )
    case = _case_from_signal(signal, prompt, answer, signal.text)
    return signal, case


def split_feedback_command(text: str) -> tuple[str | None, str]:
    if not text.startswith("/"):
        return None, text
    parts = text.split(maxsplit=1)
    command = parts[0].strip()
    payload = parts[1].strip() if len(parts) > 1 else ""
    return command, payload


def infer_context(prompt_text: str) -> str:
    lower = prompt_text.lower()
    if any(word in lower for word in ["논문", "paper", "arxiv"]):
        return "paper"
    if any(word in lower for word in ["코드", "구현", "test", "bug", "refactor"]):
        return "coding"
    if any(word in lower for word in ["삭제", "권한", "위험", "token", "secret"]):
        return "safety"
    if any(word in lower for word in ["설계", "architecture", "langgraph", "openclaw"]):
        return "architecture"
    return "general"


def _case_from_signal(
    signal: PreferenceSignal,
    prompt: ChatEvent | None,
    answer: ChatEvent | None,
    preference: str,
    preferred_text: str | None = None,
) -> ImitationCase | None:
    if prompt is None:
        return None
    return ImitationCase(
        id=f"case_{uuid.uuid4().hex}",
        source_signal_id=signal.signal_id,
        prompt=prompt.text,
        context=signal.context,
        preference=preference,
        preferred_text=preferred_text,
        avoid_text=answer.text if signal.polarity == "negative" and answer else None,
        weight=signal.strength,
    )
