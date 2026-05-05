from __future__ import annotations

import unittest

from tests.workspace import workspace_ctx
from viktor_dgmh.archive import init_workspace
from viktor_dgmh.chat import run_chat_once
from viktor_dgmh.llm import FakeProvider
from viktor_dgmh.memory import (
    append_chat_event,
    append_slack_message_map,
    extract_preference_from_feedback,
    find_slack_message_map,
    load_chat_event,
    load_imitation_cases,
    load_preferences,
    load_recent_chat_events,
)
from viktor_dgmh.models import ChatEvent


class MemoryTests(unittest.TestCase):
    def test_feedback_commands_create_preferences_and_cases(self) -> None:
        prompt = ChatEvent(event_id="p1", session_id="s1", type="prompt", role="user", text="설계 어떨까?")
        answer = ChatEvent(event_id="a1", session_id="s1", type="answer", role="agent", text="긴 배경 설명입니다.")
        feedback = ChatEvent(
            event_id="f1",
            session_id="s1",
            type="feedback",
            role="user",
            text="/too-long",
            parent_event_id="a1",
        )

        signal, case = extract_preference_from_feedback(feedback, prompt=prompt, answer=answer)

        self.assertIsNotNone(signal)
        self.assertIsNotNone(case)
        self.assertEqual(signal.kind, "too_verbose")
        self.assertEqual(signal.polarity, "negative")
        self.assertEqual(case.avoid_text, "긴 배경 설명입니다.")

    def test_rewrite_is_high_value_preferred_text(self) -> None:
        prompt = ChatEvent(event_id="p1", session_id="s1", type="prompt", role="user", text="evaluator가 뭐야?")
        answer = ChatEvent(event_id="a1", session_id="s1", type="answer", role="agent", text="장황한 설명")
        feedback = ChatEvent(
            event_id="f1",
            session_id="s1",
            type="feedback",
            role="user",
            text="/rewrite evaluator는 답을 점수화하는 심판이야.",
            parent_event_id="a1",
        )

        signal, case = extract_preference_from_feedback(feedback, prompt=prompt, answer=answer)

        self.assertEqual(signal.kind, "rewrite")
        self.assertEqual(signal.strength, 1.0)
        self.assertIn("심판", case.preferred_text)

    def test_chat_once_persists_korean_feedback(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            session_id, answer = run_chat_once(
                root,
                FakeProvider(),
                "논문은 어떻게 봐야 해?",
                feedback_text="/remember 논문은 claim/evidence/limitation으로 나눠줘",
            )

            self.assertTrue(session_id.startswith("chat_"))
            self.assertTrue(answer)
            prefs = load_preferences(root)
            cases = load_imitation_cases(root)
            self.assertEqual(len(prefs), 1)
            self.assertEqual(len(cases), 1)
            self.assertIn("claim/evidence/limitation", prefs[0].text)

    def test_slack_message_mapping_roundtrip(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            event = ChatEvent(event_id="p1", session_id="s1", type="prompt", role="user", text="설계 어떨까?")
            append_chat_event(root, event)
            append_slack_message_map(
                root,
                {
                    "channel": "C1",
                    "ts": "123.45",
                    "session_id": "s1",
                    "prompt_event_id": "p1",
                    "answer_event_id": "a1",
                },
            )

            mapping = find_slack_message_map(root, channel="C1", ts="123.45")
            loaded = load_chat_event(root, "s1", "p1")

            self.assertEqual(mapping["prompt_event_id"], "p1")
            self.assertEqual(loaded.text, "설계 어떨까?")

    def test_load_recent_chat_events(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            for index in range(3):
                append_chat_event(
                    root,
                    ChatEvent(
                        event_id=f"p{index}",
                        session_id="s1",
                        type="prompt",
                        role="user",
                        text=f"질문 {index}",
                    ),
                )

            events = load_recent_chat_events(root, limit=2)

            self.assertEqual([event.text for event in events], ["질문 1", "질문 2"])


if __name__ == "__main__":
    unittest.main()
