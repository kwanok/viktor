from __future__ import annotations

import unittest

from tests.workspace import workspace_ctx
from viktor_dgmh.archive import get_active_agent_id, init_workspace
from viktor_dgmh.auto_evolve import run_auto_evolve_once
from viktor_dgmh.chat import run_chat_once
from viktor_dgmh.llm import FakeProvider
from viktor_dgmh.memory import load_imitation_cases
from viktor_dgmh.models import Config
from viktor_dgmh.reflection import reflect_on_recent_conversation


class ReflectionAutoEvolveTests(unittest.TestCase):
    def test_reflection_creates_imitation_case_without_feedback_command(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            run_chat_once(root, FakeProvider(), "너의 코드를 고칠 수 있어?")
            run_chat_once(root, FakeProvider(), "스스로 생각해서 개선해봐")

            signals, cases = reflect_on_recent_conversation(root, FakeProvider(), use_fake=True)

            self.assertGreaterEqual(len(signals), 1)
            self.assertGreaterEqual(len(cases), 1)
            self.assertEqual(len(load_imitation_cases(root)), len(cases))

    def test_auto_evolve_reflects_then_runs_small_evolution(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            run_chat_once(root, FakeProvider(), "너의 코드를 고칠 수 있어?")
            run_chat_once(root, FakeProvider(), "스스로 생각해서 개선해봐")
            config = Config(
                auto_evolve_min_chat_events=4,
                auto_evolve_min_cases=1,
                auto_evolve_cooldown_seconds=0,
            )

            state = run_auto_evolve_once(root, config, FakeProvider(), use_fake=True)

            self.assertIsNotNone(state)
            self.assertNotEqual(get_active_agent_id(root), "gen000_seed")


if __name__ == "__main__":
    unittest.main()
