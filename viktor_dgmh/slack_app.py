from __future__ import annotations

import os
import re
from pathlib import Path

from .archive import get_active_agent_id
from .chat import answer_with_agent, record_feedback
from .llm import provider_from_config
from .memory import (
    append_chat_event,
    append_slack_message_map,
    find_slack_message_map,
    load_chat_event,
    new_event_id,
    new_session_id,
)
from .models import ChatEvent, Config

REACTION_TO_FEEDBACK = {
    "+1": "/good",
    "thumbsup": "/good",
    "-1": "/bad",
    "thumbsdown": "/bad",
    "scissors": "/too-long",
    "mag": "/weak-evidence",
    "warning": "/unsafe",
    "brain": "/remember This answer reflects a preference worth remembering.",
    "dart": "/good",
}


def serve_slack_app(root: Path, config: Config, *, use_fake: bool = False) -> None:
    try:
        from slack_bolt import App
        from slack_bolt.adapter.socket_mode import SocketModeHandler
    except ImportError as exc:
        raise RuntimeError("slack-bolt is not installed. Run `uv sync` or install dependencies.") from exc

    bot_token = os.environ.get("SLACK_BOT_TOKEN")
    app_token = os.environ.get("SLACK_APP_TOKEN")
    if not bot_token or not app_token:
        raise RuntimeError("Set SLACK_BOT_TOKEN and SLACK_APP_TOKEN before running the Slack app.")

    provider = provider_from_config(config, root=root, use_fake=use_fake)
    app = App(token=bot_token)

    @app.event("app_mention")
    def handle_app_mention(event, say, logger):
        text = _strip_bot_mentions(event.get("text", "")).strip()
        if not text:
            say(text="무엇을 도와줄지 한 줄로 말해줘.", thread_ts=event.get("thread_ts") or event.get("ts"))
            return
        _answer_and_map(root, provider, text, event, say, logger)

    @app.event("message")
    def handle_message(event, say, logger):
        if event.get("subtype") or event.get("bot_id"):
            return
        if event.get("channel_type") != "im":
            return
        text = event.get("text", "").strip()
        if not text:
            return
        _answer_and_map(root, provider, text, event, say, logger)

    @app.event("reaction_added")
    def handle_reaction(event, logger):
        item = event.get("item", {})
        channel = item.get("channel")
        ts = item.get("ts")
        reaction = event.get("reaction")
        if not channel or not ts or not reaction:
            return
        feedback = REACTION_TO_FEEDBACK.get(reaction)
        if not feedback:
            return
        mapping = find_slack_message_map(root, channel=channel, ts=ts)
        if not mapping:
            return
        prompt_event = load_chat_event(root, mapping["session_id"], mapping["prompt_event_id"])
        answer_event = load_chat_event(root, mapping["session_id"], mapping["answer_event_id"])
        if not prompt_event or not answer_event:
            return
        record_feedback(root, mapping["session_id"], feedback, prompt_event, answer_event)
        logger.info("Recorded Slack reaction feedback %s for %s/%s", reaction, channel, ts)

    print("Starting Viktor Slack app with Socket Mode.")
    SocketModeHandler(app, app_token).start()


def _answer_and_map(root: Path, provider, text: str, event: dict, say, logger) -> None:
    session_id = f"slack_{event.get('channel')}_{event.get('thread_ts') or event.get('ts') or new_session_id()}"
    prompt_event = ChatEvent(
        event_id=new_event_id("prompt"),
        session_id=session_id,
        type="prompt",
        role="user",
        text=text,
        metadata={
            "source": "slack",
            "channel": event.get("channel"),
            "ts": event.get("ts"),
            "user": event.get("user"),
        },
    )
    append_chat_event(root, prompt_event)

    try:
        answer = answer_with_agent(root, provider, text)
    except Exception as exc:
        logger.exception("Failed to answer Slack message")
        say(text=f"답변 중 오류가 났어: {exc}", thread_ts=event.get("thread_ts") or event.get("ts"))
        return

    answer_event = ChatEvent(
        event_id=new_event_id("answer"),
        session_id=session_id,
        type="answer",
        role="agent",
        text=answer,
        parent_event_id=prompt_event.event_id,
        metadata={"source": "slack", "agent_id": get_active_agent_id(root)},
    )
    append_chat_event(root, answer_event)
    response = say(text=answer, thread_ts=event.get("thread_ts") or event.get("ts"))
    append_slack_message_map(
        root,
        {
            "channel": response.get("channel") or event.get("channel"),
            "ts": response.get("ts"),
            "session_id": session_id,
            "prompt_event_id": prompt_event.event_id,
            "answer_event_id": answer_event.event_id,
        },
    )


def _strip_bot_mentions(text: str) -> str:
    return re.sub(r"<@[A-Z0-9]+>", "", text)

