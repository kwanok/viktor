from __future__ import annotations

import json
import uuid
from pathlib import Path

from .archive import load_agent
from .llm import ChatProvider
from .memory import append_capability_gap, append_imitation_case, append_preference, load_recent_chat_events
from .models import CapabilityGap, ChatEvent, ImitationCase, PreferenceSignal
from .strategy import load_reflection_policy


def reflect_on_recent_conversation(
    root: Path,
    provider: ChatProvider,
    *,
    max_events: int = 40,
    use_fake: bool = False,
) -> tuple[list[PreferenceSignal], list[ImitationCase], list[CapabilityGap]]:
    events = load_recent_chat_events(root, limit=max_events)
    if len(events) < 4:
        return [], [], []
    policy = load_reflection_policy(load_agent(root, "active").path)
    observations, gap_items = _fake_reflection(events) if use_fake else _request_reflection(events, provider, policy.model_dump())
    signals: list[PreferenceSignal] = []
    cases: list[ImitationCase] = []
    gaps: list[CapabilityGap] = []
    seen_sources = _existing_reflection_sources(root)
    seen_gaps = _existing_capability_gap_keys(root)
    pairs = _prompt_answer_pairs(events)

    for item in observations:
        source_event_id = str(item.get("source_event_id") or "")
        if not source_event_id or source_event_id in seen_sources:
            continue
        prompt, answer = pairs.get(source_event_id, (None, None))
        if prompt is None:
            continue
        target = _normalize_target(
            str(item.get("target") or "task_prompt"),
            str(item.get("context") or "general"),
            policy.self_model_contexts,
        )
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

    for item in gap_items:
        requested_capability = str(item.get("requested_capability") or "").strip()
        source = str(item.get("source") or "conversation_reflection")
        evidence = _string_list(item.get("evidence"))
        key = (requested_capability, tuple(evidence))
        if not requested_capability or key in seen_gaps:
            continue
        gap = CapabilityGap(
            gap_id=f"gap_{uuid.uuid4().hex}",
            source=source,
            summary=str(item.get("summary") or "Capability gap found from conversation."),
            evidence=evidence,
            requested_capability=requested_capability,
            failure_mode=str(item.get("failure_mode") or "claimed_or_implied_missing_capability"),
            required_changes=_string_list(item.get("required_changes")),
            requires_restart=bool(item.get("requires_restart", False)),
            status=_normalize_gap_status(str(item.get("status") or "open")),  # type: ignore[arg-type]
        )
        append_capability_gap(root, gap)
        gaps.append(gap)
    return signals, cases, gaps


def _request_reflection(events: list[ChatEvent], provider: ChatProvider, policy: dict) -> tuple[list[dict], list[dict]]:
    transcript = "\n".join(f"{event.event_id} {event.role}: {event.text}" for event in events)
    prompt = (
        "Review this user/agent transcript for self-improvement opportunities. "
        "Follow the active reflection policy, then infer problems from the transcript. "
        "Return JSON only with keys `observations` and `capability_gaps`. "
        "`observations` contains preference/behavior observations with source_event_id, "
        "target (self_model|task_prompt|policy), kind, polarity (positive|negative|neutral), strength (0..1), "
        "context, preference, and optional preferred_text. "
        "`capability_gaps` contains missing runtime/code/tool capabilities with source, summary, evidence, "
        "requested_capability, failure_mode, required_changes, requires_restart, and status. "
        "Do not treat missing Slack actions as prompt/style issues when they require scopes, code, or restart.\n\n"
        f"Active reflection policy:\n{json.dumps(policy, ensure_ascii=False, indent=2)}\n\n"
        f"{transcript}"
    )
    raw = provider.chat([{"role": "user", "content": prompt}], response_format="json")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [], []
    observations = data.get("observations", [])
    gaps = data.get("capability_gaps", [])
    return observations if isinstance(observations, list) else [], gaps if isinstance(gaps, list) else []


def _fake_reflection(events: list[ChatEvent]) -> tuple[list[dict], list[dict]]:
    observations = []
    gaps = []
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
    if _mentions_slack_eyes_reaction_gap(events):
        evidence = [event.text for event in events if ":eyes:" in event.text or "이모지" in event.text or "리액션" in event.text]
        gaps.append(
            {
                "source": "conversation_reflection",
                "summary": "User asked Viktor to add :eyes: as a Slack reaction to the user's message, but Viktor lacks that runtime capability.",
                "evidence": evidence[-4:],
                "requested_capability": "slack_reaction_add",
                "failure_mode": "missing_slack_action_capability",
                "required_changes": ["slack_scope", "code", "restart"],
                "requires_restart": True,
                "status": "open",
            }
        )
    return observations, gaps


def _normalize_target(target: str, context: str, self_model_contexts: list[str] | None = None) -> str:
    if context in set(self_model_contexts or ["identity", "korean_style", "relationship", "tone"]):
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


def _mentions_slack_eyes_reaction_gap(events: list[ChatEvent]) -> bool:
    texts = [event.text for event in events if event.role == "user"]
    has_eyes = any(":eyes:" in text or "eyes" in text.lower() for text in texts)
    asks_reaction = any("이모지" in text or "리액션" in text or "reaction" in text.lower() for text in texts)
    asks_to_try = any("해봐" in text or "달아" in text or "붙" in text for text in texts)
    return has_eyes and asks_reaction and asks_to_try


def _string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _normalize_gap_status(value: str) -> str:
    return value if value in {"open", "planned", "resolved", "dismissed"} else "open"


def _existing_reflection_sources(root: Path) -> set[str]:
    from .memory import load_preferences

    return {signal.source_event_id for signal in load_preferences(root) if signal.kind.startswith("reflection")}


def _existing_capability_gap_keys(root: Path) -> set[tuple[str, tuple[str, ...]]]:
    from .memory import load_capability_gaps

    return {(gap.requested_capability, tuple(gap.evidence)) for gap in load_capability_gaps(root)}
