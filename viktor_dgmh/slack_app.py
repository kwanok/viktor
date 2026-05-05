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
from .models import ChatEvent, Config, RouterObservation
from .router import add_router_label_for_slack_message, append_router_observation, load_router_policy, score_message
from .shell_runner import (
    format_shell_result,
    parse_shell_command,
    run_shell_command,
    shell_enabled,
    shell_user_allowed,
)

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

    if not os.environ.get("MODEL_PROVIDER") and not os.environ.get("OPENAI_API_KEY"):
        config.model_provider = "codex_cli"
    provider = provider_from_config(config, root=root, use_fake=use_fake)
    app = App(token=bot_token)

    @app.event("app_mention")
    def handle_app_mention(event, say, logger):
        text = _strip_bot_mentions(event.get("text", "")).strip()
        if not text:
            say(text="What should I help with?", thread_ts=event.get("thread_ts") or event.get("ts"))
            return
        if parse_shell_command(text):
            _handle_shell_command(root, config, text, event, say, logger)
            return
        _answer_and_map(root, provider, text, event, say, logger)

    @app.event("message")
    def handle_message(event, say, logger):
        if event.get("subtype") or event.get("bot_id"):
            return
        if event.get("channel_type") != "im":
            if event.get("channel_type") in {"channel", "group"}:
                _handle_channel_message(root, provider, config, event, say, logger)
            return
        text = event.get("text", "").strip()
        if not text:
            return
        if parse_shell_command(text):
            _handle_shell_command(root, config, text, event, say, logger)
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
        router_label = add_router_label_for_slack_message(root, channel=channel, slack_ts=ts, reaction=reaction)
        if router_label:
            logger.info("Recorded router label %s for %s/%s", router_label.label, channel, ts)
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


def _handle_channel_message(root: Path, provider, config: Config, event: dict, say, logger) -> None:
    text = event.get("text", "").strip()
    if not text:
        return
    if parse_shell_command(text):
        _handle_shell_command(root, config, text, event, say, logger)
        return

    enabled = _env_bool("SLACK_AUTO_RESPOND_CHANNELS", config.slack_auto_respond_channels)
    if not enabled:
        return
    policy = load_router_policy(root, "active")
    decision = score_message(text, policy)
    if decision.score < config.slack_min_respond_score:
        decision.should_respond = False
    _log_channel_observation(root, event, decision)
    if not decision.should_respond:
        logger.debug("Staying silent in channel: %s", decision.reason)
        return
    logger.info("Responding in channel with score %.2f: %s", decision.score, decision.reason)
    _answer_and_map(root, provider, text, event, say, logger)


def _handle_shell_command(root: Path, config: Config, text: str, event: dict, say, logger) -> None:
    command = parse_shell_command(text)
    thread_ts = event.get("thread_ts") or event.get("ts")
    if not command:
        return
    if not shell_enabled(config):
        say(
            text="Shell command is disabled. Set `SLACK_ENABLE_SHELL=true` to enable it.",
            thread_ts=thread_ts,
        )
        return
    if not shell_user_allowed(event.get("user")):
        say(
            text="Shell command denied. Add your Slack user id to `SLACK_SHELL_ALLOWED_USERS`.",
            thread_ts=thread_ts,
        )
        return
    try:
        record = run_shell_command(
            root,
            command,
            config=config,
            user=event.get("user"),
            channel=event.get("channel"),
            slack_ts=event.get("ts"),
        )
    except Exception as exc:
        logger.exception("Failed to run Slack shell command")
        say(text=f"Shell command failed before execution: {exc}", thread_ts=thread_ts)
        return
    logger.info("Ran Slack shell command %s with exit code %s", record.command_id, record.exit_code)
    say(text=format_shell_result(record), thread_ts=thread_ts)


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
        say(text=f"Answer failed: {exc}", thread_ts=event.get("thread_ts") or event.get("ts"))
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


def should_respond_to_channel_message(text: str, *, min_score: float = 0.65):
    decision = score_message(text)
    if decision.score < min_score:
        decision.should_respond = False
    return decision


def _log_channel_observation(root: Path, event: dict, decision) -> None:
    observation_id = new_event_id("robs")
    append_router_observation(
        root,
        RouterObservation(
            observation_id=observation_id,
            channel=event.get("channel", ""),
            slack_ts=event.get("ts", ""),
            user=event.get("user"),
            text=event.get("text", ""),
            router_id=decision.router_id,
            should_respond=decision.should_respond,
            score=decision.score,
            reason=decision.reason,
        ),
    )
    session_id = f"slack_observe_{event.get('channel')}_{event.get('ts')}"
    observation = ChatEvent(
        event_id=observation_id,
        session_id=session_id,
        type="prompt",
        role="user",
        text=event.get("text", ""),
        metadata={
            "source": "slack_channel_observation",
            "channel": event.get("channel"),
            "ts": event.get("ts"),
            "user": event.get("user"),
            "should_respond": decision.should_respond,
            "respond_score": decision.score,
            "respond_reason": decision.reason,
            "router_id": decision.router_id,
        },
    )
    append_chat_event(root, observation)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}
