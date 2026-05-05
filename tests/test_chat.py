from __future__ import annotations

import unittest

from tests.workspace import workspace_ctx
from viktor_dgmh.archive import init_workspace
from viktor_dgmh.chat import answer_with_agent
from viktor_dgmh.memory import append_preference
from viktor_dgmh.models import PreferenceSignal


class CapturingProvider:
    def __init__(self) -> None:
        self.messages = []

    def chat(self, messages, *, model=None, response_format=None) -> str:
        self.messages = messages
        return "나는 빅토르야."


class ChatTests(unittest.TestCase):
    def test_answer_prompt_includes_recent_learned_preferences(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            append_preference(
                root,
                PreferenceSignal(
                    signal_id="s1",
                    session_id="chat1",
                    source_event_id="p1",
                    kind="reflection_identity",
                    polarity="negative",
                    strength=0.95,
                    context="identity",
                    target="self_model",
                    text="Outwardly be Viktor only; do not describe yourself as a task agent.",
                    preferred_text="나는 빅토르야.",
                ),
            )
            provider = CapturingProvider()

            answer = answer_with_agent(root, provider, "너는 누구야?")

            self.assertEqual(answer, "나는 빅토르야.")
            system_prompt = provider.messages[0]["content"]
            self.assertIn("Self model:", system_prompt)
            self.assertIn("Public name: Viktor", system_prompt)
            self.assertIn("Forbidden outward self-descriptions", system_prompt)
            self.assertIn("Recent learned user preferences", system_prompt)
            self.assertIn("Outwardly be Viktor only", system_prompt)
            self.assertIn("나는 빅토르야.", system_prompt)


if __name__ == "__main__":
    unittest.main()
