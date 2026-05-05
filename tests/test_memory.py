from __future__ import annotations

import unittest

from tests.workspace import workspace_ctx
from viktor_dgmh.archive import init_workspace
from viktor_dgmh.chat import run_chat_once
from viktor_dgmh.llm import FakeProvider
from viktor_dgmh.memory import (
    extract_preference_from_feedback,
    load_imitation_cases,
    load_preferences,
)
from viktor_dgmh.models import ChatEvent


class MemoryTests(unittest.TestCase):
    def test_feedback_commands_create_preferences_and_cases(self) -> None:
        prompt = ChatEvent(event_id="p1", session_id="s1", type="prompt", role="user", text="이 설계 어때?")
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
            text="/rewrite evaluator는 답변을 점수화하는 심판이야.",
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
            self.assertIn("결론", answer)
            prefs = load_preferences(root)
            cases = load_imitation_cases(root)
            self.assertEqual(len(prefs), 1)
            self.assertEqual(len(cases), 1)
            self.assertIn("claim/evidence/limitation", prefs[0].text)


if __name__ == "__main__":
    unittest.main()

