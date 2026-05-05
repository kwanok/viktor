from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from tests.workspace import workspace_ctx
from viktor_dgmh.archive import init_workspace
from viktor_dgmh.llm import FakeProvider
from viktor_dgmh.memory import load_capability_gaps
from viktor_dgmh.models import Config
from viktor_dgmh.slack_app import (
    REACTION_TO_FEEDBACK,
    _answer_and_map,
    _compose_slack_prompt,
    _handle_channel_message,
    _has_slack_user_mention,
    _strip_bot_mentions,
    _thread_context_from_slack,
    should_respond_to_channel_message,
)


class FakeSlackClient:
    def __init__(self, messages: list[dict[str, str]]) -> None:
        self.messages = messages

    def conversations_replies(self, **kwargs):
        return {"messages": self.messages}


class FakeLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def debug(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        raise AssertionError("logger.exception should not be called")


class CapturingProvider:
    def __init__(self) -> None:
        self.messages = []

    def chat(self, messages, *, model=None, response_format=None):
        self.messages = messages
        return "gap recorded"


def fake_say(**kwargs):
    return {"channel": "C1", "ts": "2"}


class CountingSay:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"channel": "C1", "ts": "2"}


class SlackAppTests(unittest.TestCase):
    def test_reaction_mapping_matches_feedback_commands(self) -> None:
        self.assertEqual(REACTION_TO_FEEDBACK["scissors"], "/too-long")
        self.assertEqual(REACTION_TO_FEEDBACK["mag"], "/weak-evidence")
        self.assertEqual(REACTION_TO_FEEDBACK["warning"], "/unsafe")

    def test_strip_bot_mentions(self) -> None:
        self.assertEqual(_strip_bot_mentions("<@U123> evaluator 설명해줘").strip(), "evaluator 설명해줘")

    def test_has_slack_user_mention(self) -> None:
        self.assertTrue(_has_slack_user_mention("<@U123> 안녕"))
        self.assertFalse(_has_slack_user_mention("빅토르 안녕"))

    def test_channel_router_responds_to_judgment_question(self) -> None:
        decision = should_respond_to_channel_message("이 LangGraph 설계 괜찮을까?", min_score=0.65)

        self.assertTrue(decision.should_respond)
        self.assertGreaterEqual(decision.score, 0.65)

    def test_channel_router_stays_silent_for_chatter(self) -> None:
        decision = should_respond_to_channel_message("ㅋㅋ 그냥 잡담", min_score=0.65)

        self.assertFalse(decision.should_respond)

    def test_channel_router_responds_to_viktor_alias(self) -> None:
        decision = should_respond_to_channel_message("빅토르 선생님 뭐하시나요?", min_score=0.65)

        self.assertTrue(decision.should_respond)

    def test_channel_handler_skips_mention_events(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            say = CountingSay()

            _handle_channel_message(
                root,
                FakeProvider(),
                Config(),
                {"channel": "C1", "ts": "1", "user": "U1", "text": "<@BOT> 안녕"},
                say,
                FakeLogger(),
            )

            self.assertEqual(say.calls, [])

    def test_thread_context_skips_current_message(self) -> None:
        context = _thread_context_from_slack(
            FakeSlackClient(
                [
                    {"ts": "1", "user": "U1", "text": "<@BOT> GPU가 뭐야?"},
                    {"ts": "2", "bot_id": "B1", "text": "GPU access blocked."},
                    {"ts": "3", "user": "U1", "text": "우회해서 확인해봐"},
                ]
            ),
            {"channel": "C1", "thread_ts": "1", "ts": "3"},
            FakeLogger(),
        )

        self.assertIn("user:U1: GPU가 뭐야?", context or "")
        self.assertIn("assistant: GPU access blocked.", context or "")
        self.assertNotIn("우회해서 확인해봐", context or "")

    def test_compose_slack_prompt_includes_thread_context(self) -> None:
        prompt = _compose_slack_prompt("왜 그래?", "user:U1: 이전 질문\nassistant: 이전 답")

        self.assertIn("Slack thread context", prompt)
        self.assertIn("Current Slack message", prompt)
        self.assertIn("왜 그래?", prompt)

    def test_answer_schedules_conversation_evolution(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            with patch("viktor_dgmh.slack_app.schedule_auto_evolve_after_conversation") as schedule:
                _answer_and_map(
                    root,
                    FakeProvider(),
                    "evaluator가 뭐야?",
                    {"channel": "C1", "ts": "1", "user": "U1"},
                    fake_say,
                    FakeLogger(),
                    config=Config(auto_evolve_min_chat_events=1),
                    use_fake=True,
                )

            self.assertTrue(schedule.called)

    def test_answer_records_slack_reaction_capability_gap_from_thread_context(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            provider = CapturingProvider()
            thread_context = (
                "user:U1: Please add :eyes: as a reaction to my Slack message.\n"
                "assistant: I cannot add Slack reactions yet."
            )

            with patch("viktor_dgmh.slack_app.schedule_auto_evolve_after_conversation"):
                _answer_and_map(
                    root,
                    provider,
                    "그냥 해줘",
                    {"channel": "C1", "ts": "3", "thread_ts": "1", "user": "U1"},
                    fake_say,
                    FakeLogger(),
                    config=Config(auto_evolve_min_chat_events=1),
                    thread_context=thread_context,
                    use_fake=True,
                )

            gaps = load_capability_gaps(root)
            self.assertEqual(len(gaps), 1)
            self.assertEqual(gaps[0].requested_capability, "slack_reaction_add")
            self.assertEqual(gaps[0].required_changes, ["slack_scope", "code", "restart"])
            self.assertTrue(gaps[0].requires_restart)
            joined_prompt = "\n".join(message["content"] for message in provider.messages)
            self.assertIn("capability gap was recorded", joined_prompt)

    def test_slack_reaction_capability_gap_is_not_duplicated(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            thread_context = "user:U1: Please add :eyes: as a reaction to my Slack message."

            with patch("viktor_dgmh.slack_app.schedule_auto_evolve_after_conversation"):
                _answer_and_map(
                    root,
                    CapturingProvider(),
                    "make it work",
                    {"channel": "C1", "ts": "3", "thread_ts": "1", "user": "U1"},
                    fake_say,
                    FakeLogger(),
                    config=Config(auto_evolve_min_chat_events=1),
                    thread_context=thread_context,
                    use_fake=True,
                )
                _answer_and_map(
                    root,
                    CapturingProvider(),
                    "make it work",
                    {"channel": "C1", "ts": "4", "thread_ts": "1", "user": "U1"},
                    fake_say,
                    FakeLogger(),
                    config=Config(auto_evolve_min_chat_events=1),
                    thread_context=thread_context,
                    use_fake=True,
                )

            self.assertEqual(len(load_capability_gaps(root)), 1)

    def test_manifest_does_not_grant_reaction_write_scope_yet(self) -> None:
        manifest = yaml.safe_load((Path.cwd() / "slack_app_manifest.yaml").read_text(encoding="utf-8"))
        scopes = manifest["oauth_config"]["scopes"]["bot"]

        self.assertNotIn("reactions:write", scopes)


if __name__ == "__main__":
    unittest.main()
