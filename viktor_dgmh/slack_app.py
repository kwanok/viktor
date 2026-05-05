from __future__ import annotations

import os
import re
from pathlib import Path

from .archive import get_active_agent_id
from .auto_evolve import schedule_auto_evolve_after_conversation
from .chat import answer_with_agent, record_feedback
from .llm import provider_from_config
from .memory import (
    append_capability_gap,
    append_chat_event,
    append_slack_message_map,
    find_slack_message_map,
    load_capability_gaps,
    load_chat_event,
    new_event_id,
    new_session_id,
)
from .models import CapabilityGap, ChatEvent, Config, RouterObservation
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
    def handle_app_mention(event, say, logger, client):
        text = _strip_bot_mentions(event.get("text", "")).strip()
        if not text:
            say(text="What should I help with?", thread_ts=event.get("thread_ts") or event.get("ts"))
            return
        if parse_shell_command(text):
            _handle_shell_command(root, config, text, event, say, logger)
            return
        thread_context = _thread_context_from_slack(client, event, logger)
        _answer_and_map(
            root,
            provider,
            text,
            event,
            say,
            logger,
            config=config,
            thread_context=thread_context,
            use_fake=use_fake,
        )

    @app.event("message")
    def handle_message(event, say, logger, client):
        if event.get("subtype") or event.get("bot_id"):
            return
        if event.get("channel_type") != "im":
            if event.get("channel_type") in {"channel", "group"}:
                _handle_channel_message(root, provider, config, event, say, logger, client, use_fake=use_fake)
            return
        text = event.get("text", "").strip()
        if not text:
            return
        if parse_shell_command(text):
            _handle_shell_command(root, config, text, event, say, logger)
            return
        thread_context = _thread_context_from_slack(client, event, logger)
        _answer_and_map(
            root,
            provider,
            text,
            event,
            say,
            logger,
            config=config,
            thread_context=thread_context,
            use_fake=use_fake,
        )

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
        schedule_auto_evolve_after_conversation(
            root,
            config,
            provider,
            use_fake=use_fake,
            reason=f"slack_reaction:{reaction}",
            logger=logger,
        )

    print("Starting Viktor Slack app with Socket Mode.")
    SocketModeHandler(app, app_token).start()


def _handle_channel_message(
    root: Path,
    provider,
    config: Config,
    event: dict,
    say,
    logger,
    client=None,
    *,
    use_fake: bool = False,
) -> None:
    text = event.get("text", "").strip()
    if not text:
        return
    if _has_slack_user_mention(text):
        logger.debug("Skipping channel message with Slack mention; app_mention handles direct mentions.")
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
    thread_context = _thread_context_from_slack(client, event, logger)
    _answer_and_map(
        root,
        provider,
        text,
        event,
        say,
        logger,
        config=config,
        thread_context=thread_context,
        use_fake=use_fake,
    )


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


def _thread_context_from_slack(client, event: dict, logger, *, max_messages: int = 12) -> str | None:
    if client is None:
        return None
    channel = event.get("channel")
    thread_ts = event.get("thread_ts")
    current_ts = event.get("ts")
    if not channel or not thread_ts:
        return None
    try:
        response = client.conversations_replies(channel=channel, ts=thread_ts, limit=max_messages, inclusive=True)
    except Exception:
        logger.exception("Failed to fetch Slack thread context")
        return None

    messages = response.get("messages", [])
    lines = []
    for message in messages[-max_messages:]:
        if message.get("ts") == current_ts:
            continue
        text = _strip_bot_mentions(message.get("text", "")).strip()
        if not text:
            continue
        speaker = "assistant" if message.get("bot_id") else f"user:{message.get('user', 'unknown')}"
        lines.append(f"{speaker}: {text}")
    if not lines:
        return None
    return "\n".join(lines)


def _compose_slack_prompt(text: str, thread_context: str | None) -> str:
    if not thread_context:
        return text
    return (
        "Slack thread context, oldest to newest:\n"
        f"{thread_context}\n\n"
        "Current Slack message:\n"
        f"{text}\n\n"
        "Answer the current message using the thread context. "
        "If the current message refers to previous messages, resolve that reference from the context."
    )


def _record_slack_capability_gap_if_needed(
    root: Path,
    text: str,
    thread_context: str | None,
    event: dict,
    logger,
) -> CapabilityGap | None:
    if not _looks_like_slack_reaction_gap(text, thread_context):
        return None

    for gap in load_capability_gaps(root):
        if (
            gap.status in {"open", "planned"}
            and gap.requested_capability == "slack_reaction_add"
            and gap.failure_mode == "missing_slack_action_capability"
        ):
            return gap

    gap = CapabilityGap(
        gap_id=new_event_id("gap"),
        source="slack_message",
        summary=(
            "User asked Viktor to add :eyes: as a Slack reaction to the user's message, "
            "but the runtime cannot perform that Slack action yet."
        ),
        evidence=_slack_capability_gap_evidence(text, thread_context, event),
        requested_capability="slack_reaction_add",
        failure_mode="missing_slack_action_capability",
        required_changes=["slack_scope", "code", "restart"],
        requires_restart=True,
    )
    append_capability_gap(root, gap)
    if logger:
        logger.info("Recorded Slack capability gap %s for %s", gap.gap_id, gap.requested_capability)
    return gap


def _looks_like_slack_reaction_gap(text: str, thread_context: str | None) -> bool:
    combined = "\n".join(part for part in [thread_context or "", text] if part)
    lowered = combined.lower()
    has_eyes = ":eyes:" in lowered or "👀" in combined or "eyes" in lowered
    has_reaction_word = (
        "reaction" in lowered
        or "react" in lowered
        or "emoji" in lowered
        or any(word in combined for word in ["리액션", "이모지", "반응"])
    )
    asks_for_action = any(
        word in combined
        for word in ["달아", "붙", "해봐", "해줘", "되게", "추가", "하도록", "실제로", "그냥", "add", "make it work"]
    )
    return has_eyes and has_reaction_word and asks_for_action


def _slack_capability_gap_evidence(text: str, thread_context: str | None, event: dict) -> list[str]:
    evidence = [
        f"current Slack message: {_truncate_evidence(text)}",
        f"channel={event.get('channel')} ts={event.get('ts')} thread_ts={event.get('thread_ts')}",
    ]
    if thread_context:
        relevant_lines = [
            line
            for line in thread_context.splitlines()
            if _looks_like_slack_reaction_gap("", line) or ":eyes:" in line.lower() or "👀" in line
        ]
        for line in relevant_lines[-4:]:
            evidence.append(f"thread context: {_truncate_evidence(line)}")
    return evidence


def _capability_gap_prompt_note(gap: CapabilityGap) -> str:
    return (
        "System note before answering: a capability gap was recorded for "
        f"{gap.requested_capability}. Do not claim the Slack action is implemented or already done. "
        "Answer briefly that the gap is now recorded as self-evolution input, and keep the actual "
        "Slack reaction capability as future work."
    )


def _truncate_evidence(text: str, limit: int = 500) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _answer_and_map(
    root: Path,
    provider,
    text: str,
    event: dict,
    say,
    logger,
    *,
    config: Config,
    thread_context: str | None = None,
    use_fake: bool = False,
) -> None:
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
            "thread_ts": event.get("thread_ts"),
            "thread_context": thread_context,
        },
    )
    append_chat_event(root, prompt_event)

    try:
        prompt_text = _compose_slack_prompt(text, thread_context)
        gap = _record_slack_capability_gap_if_needed(root, text, thread_context, event, logger)
        if gap:
            prompt_text = f"{_capability_gap_prompt_note(gap)}\n\n{prompt_text}"
        answer = answer_with_agent(root, provider, prompt_text)
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
    schedule_auto_evolve_after_conversation(
        root,
        config,
        provider,
        use_fake=use_fake,
        reason="slack_conversation",
        logger=logger,
    )


def _strip_bot_mentions(text: str) -> str:
    return re.sub(r"<@[A-Z0-9]+>", "", text)


def _has_slack_user_mention(text: str) -> bool:
    return bool(re.search(r"<@[A-Z0-9]+>", text))


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
