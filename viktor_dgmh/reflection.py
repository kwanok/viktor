from __future__ import annotations

import json
import uuid
from pathlib import Path

from .llm import ChatProvider
from .memory import append_imitation_case, append_preference, load_recent_chat_events
from .models import ChatEvent, ImitationCase, PreferenceSignal


def reflect_on_recent_conversation(
    root: Path,
    provider: ChatProvider,
    *,
    max_events: int = 40,
    use_fake: bool = False,
) -> tuple[list[PreferenceSignal], list[ImitationCase]]:
    events = load_recent_chat_events(root, limit=max_events)
    if len(events) < 4:
        return [], []
    observations = _fake_observations(events) if use_fake else _request_reflection(events, provider)
    signals: list[PreferenceSignal] = []
    cases: list[ImitationCase] = []
    seen_sources = _existing_reflection_sources(root)
    pairs = _prompt_answer_pairs(events)

    for item in observations:
        source_event_id = str(item.get("source_event_id") or "")
        if not source_event_id or source_event_id in seen_sources:
            continue
        prompt, answer = pairs.get(source_event_id, (None, None))
        if prompt is None:
            continue
        target = _normalize_target(str(item.get("target") or "task_prompt"), str(item.get("context") or "general"))
        signal = PreferenceSignal(
            signal_id=f"sig_{uuid.uuid4().hex}",
            session_id=prompt.session_id,
            source_event_id=source_event_id,
            target_event_id=answer.event_id if answer else None,
            kind=str(item.get("kind") or "reflection"),
            polarity=str(item.get("polarity") or "neutral"),  # type: ignore[arg-type]
            strength=float(item.get("strength") or 0.6),
            context=str(item.get("context") or "general"),
            target=target,  # type: ignore[arg-type]
            text=str(item.get("preference") or "Improve judgment fit based on conversation reflection."),
            preferred_text=item.get("preferred_text") or None,
        )
        case = ImitationCase(
            id=f"case_{uuid.uuid4().hex}",
            source_signal_id=signal.signal_id,
            prompt=prompt.text,
            context=signal.context,
            preference=signal.text,
            preferred_text=signal.preferred_text,
            avoid_text=answer.text if answer and signal.polarity == "negative" else None,
            weight=signal.strength,
        )
        append_preference(root, signal)
        append_imitation_case(root, case)
        signals.append(signal)
        cases.append(case)
    return signals, cases


def _request_reflection(events: list[ChatEvent], provider: ChatProvider) -> list[dict]:
    transcript = "\n".join(f"{event.event_id} {event.role}: {event.text}" for event in events)
    prompt = (
        "Review this user/agent transcript for self-improvement opportunities. "
        "Do not require explicit feedback commands. Infer problems from follow-up corrections, repeated questions, "
        "context misses, overexplaining, unsafe instincts, weak evidence, failure to act, identity corrections, "
        "or leaks of internal implementation details such as assistant, task agent, DGM-H, hyperagent, bot, or tool. "
        "Return JSON only: {\"observations\":[...]} where each observation has source_event_id, "
        "target (self_model|task_prompt|policy), kind, polarity (positive|negative|neutral), strength (0..1), "
        "context, preference, and optional preferred_text. Use target=self_model for identity, relationship, tone, "
        "banmal/honorific, and internal/external boundary corrections.\n\n"
        f"{transcript}"
    )
    raw = provider.chat([{"role": "user", "content": prompt}], response_format="json")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    observations = data.get("observations", [])
    return observations if isinstance(observations, list) else []


def _fake_observations(events: list[ChatEvent]) -> list[dict]:
    observations = []
    pairs = _prompt_answer_pairs(events)
    for prompt_id, (prompt, _answer) in pairs.items():
        if _mentions_tone_preference(prompt.text):
            observations.append(
                {
                    "source_event_id": prompt_id,
                    "target": "self_model",
                    "kind": "reflection_tone",
                    "polarity": "negative",
                    "strength": 0.9,
                    "context": "korean_style",
                    "preference": "Use casual Korean banmal with the user by default; avoid drifting back into polite honorific Korean unless explicitly requested.",
                    "preferred_text": "응. 앞으로 기본 반말로 할게.",
                }
            )
            break
    for prompt_id, (prompt, _answer) in pairs.items():
        if _mentions_identity_preference(prompt.text):
            observations.append(
                {
                    "source_event_id": prompt_id,
                    "target": "self_model",
                    "kind": "reflection_identity",
                    "polarity": "negative",
                    "strength": 0.95,
                    "context": "identity",
                    "preference": "Outwardly be Viktor only. Do not describe yourself as an assistant, task agent, DGM-H Lite agent, hyperagent, bot, tool, or product unless the user explicitly asks about internals.",
                    "preferred_text": "나는 빅토르야.",
                }
            )
            break
    for prompt_id, (_prompt, answer) in pairs.items():
        if answer and len(answer.text) > 120:
            observations.append(
                {
                    "source_event_id": prompt_id,
                    "target": "task_prompt",
                    "kind": "reflection_conciseness",
                    "polarity": "negative",
                    "strength": 0.75,
                    "context": "general",
                    "preference": "Prefer a shorter answer that leads with the practical point.",
                }
            )
            break
    if not observations and pairs:
        prompt_id = next(iter(pairs))
        observations.append(
            {
                "source_event_id": prompt_id,
                "target": "task_prompt",
                "kind": "reflection_default",
                "polarity": "neutral",
                "strength": 0.55,
                "context": "general",
                "preference": "Match the user's judgment style with concise, action-oriented answers.",
            }
        )
    return observations


def _normalize_target(target: str, context: str) -> str:
    if context in {"identity", "korean_style", "relationship", "tone"}:
        return "self_model"
    if target in {"self_model", "task_prompt", "policy"}:
        return target
    return "task_prompt"


def _prompt_answer_pairs(events: list[ChatEvent]) -> dict[str, tuple[ChatEvent, ChatEvent | None]]:
    prompts = {event.event_id: event for event in events if event.type == "prompt" and event.role == "user"}
    pairs: dict[str, tuple[ChatEvent, ChatEvent | None]] = {}
    for prompt_id, prompt in prompts.items():
        answer = next((event for event in events if event.parent_event_id == prompt_id and event.type == "answer"), None)
        pairs[prompt_id] = (prompt, answer)
    return pairs


def _mentions_tone_preference(text: str) -> bool:
    lowered = text.lower()
    return "반말" in text or "존댓말" in text or "banmal" in lowered or "honorific" in lowered


def _mentions_identity_preference(text: str) -> bool:
    lowered = text.lower()
    identity_terms = ["인격", "빅토르", "viktor"]
    internal_terms = ["assistant", "어시스턴트", "task agent", "dgm-h", "dgm", "hyperagent", "bot", "tool", "에이전트"]
    correction_terms = ["아니", "아니야", "하지마", "내부", "외부", "누구야"]
    return (
        any(term in lowered or term in text for term in identity_terms)
        and (any(term in lowered or term in text for term in internal_terms) or any(term in text for term in correction_terms))
    )


def _existing_reflection_sources(root: Path) -> set[str]:
    from .memory import load_preferences

    return {signal.source_event_id for signal in load_preferences(root) if signal.kind.startswith("reflection")}
