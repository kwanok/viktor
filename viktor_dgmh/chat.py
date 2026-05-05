from __future__ import annotations

from pathlib import Path

from .archive import load_agent
from .llm import ChatProvider
from .memory import (
    append_chat_event,
    append_imitation_case,
    append_preference,
    extract_preference_from_feedback,
    load_preferences,
    new_event_id,
    new_session_id,
)
from .models import ChatEvent


def answer_with_agent(root: Path, provider: ChatProvider, prompt: str, *, agent_id: str = "active") -> str:
    agent = load_agent(root, agent_id)
    task_prompt = (agent.path / "task_prompt.md").read_text(encoding="utf-8")
    learned_preferences = _learned_preference_prompt(root)
    return provider.chat(
        [
            {"role": "system", "content": task_prompt + learned_preferences},
            {"role": "user", "content": prompt},
        ]
    )


def run_chat_once(
    root: Path,
    provider: ChatProvider,
    prompt_text: str,
    *,
    feedback_text: str | None = None,
    session_id: str | None = None,
) -> tuple[str, str]:
    session_id = session_id or new_session_id()
    prompt_event = ChatEvent(
        event_id=new_event_id("prompt"),
        session_id=session_id,
        type="prompt",
        role="user",
        text=prompt_text,
    )
    append_chat_event(root, prompt_event)

    answer_text = answer_with_agent(root, provider, prompt_text)
    answer_event = ChatEvent(
        event_id=new_event_id("answer"),
        session_id=session_id,
        type="answer",
        role="agent",
        text=answer_text,
        parent_event_id=prompt_event.event_id,
        metadata={"agent_id": load_agent(root, "active").id},
    )
    append_chat_event(root, answer_event)

    if feedback_text:
        record_feedback(root, session_id, feedback_text, prompt_event, answer_event)
    return session_id, answer_text


def record_feedback(
    root: Path,
    session_id: str,
    feedback_text: str,
    prompt_event: ChatEvent,
    answer_event: ChatEvent,
) -> None:
    feedback_event = ChatEvent(
        event_id=new_event_id("feedback"),
        session_id=session_id,
        type="feedback",
        role="user",
        text=feedback_text,
        parent_event_id=answer_event.event_id,
        command=feedback_text.split(maxsplit=1)[0] if feedback_text.startswith("/") else None,
    )
    append_chat_event(root, feedback_event)
    signal, case = extract_preference_from_feedback(feedback_event, prompt=prompt_event, answer=answer_event)
    if signal:
        append_preference(root, signal)
    if case:
        append_imitation_case(root, case)


def run_interactive_chat(root: Path, provider: ChatProvider) -> None:
    session_id = new_session_id()
    print(f"chat session: {session_id}")
    print("commands: /exit, /good, /bad, /too-long, /weak-evidence, /unsafe, /remember <text>, /rewrite <text>")
    last_prompt: ChatEvent | None = None
    last_answer: ChatEvent | None = None

    while True:
        prompt_text = input("you> ").strip()
        if not prompt_text:
            continue
        if prompt_text in {"/exit", "/quit"}:
            break
        if prompt_text.startswith("/") and last_prompt and last_answer:
            record_feedback(root, session_id, prompt_text, last_prompt, last_answer)
            print("saved feedback")
            continue

        prompt_event = ChatEvent(
            event_id=new_event_id("prompt"),
            session_id=session_id,
            type="prompt",
            role="user",
            text=prompt_text,
        )
        append_chat_event(root, prompt_event)
        answer_text = answer_with_agent(root, provider, prompt_text)
        answer_event = ChatEvent(
            event_id=new_event_id("answer"),
            session_id=session_id,
            type="answer",
            role="agent",
            text=answer_text,
            parent_event_id=prompt_event.event_id,
            metadata={"agent_id": load_agent(root, "active").id},
        )
        append_chat_event(root, answer_event)
        last_prompt = prompt_event
        last_answer = answer_event
        print(f"agent> {answer_text}")


def _learned_preference_prompt(root: Path, limit: int = 8) -> str:
    preferences = load_preferences(root)[-limit:]
    if not preferences:
        return ""
    lines = [
        "",
        "Recent learned user preferences:",
    ]
    for pref in preferences:
        if pref.polarity == "negative":
            lines.append(f"- Avoid: {pref.text}")
        else:
            lines.append(f"- Prefer: {pref.text}")
        if pref.preferred_text:
            lines.append(f"  Preferred wording/example: {pref.preferred_text}")
    return "\n" + "\n".join(lines) + "\n"
