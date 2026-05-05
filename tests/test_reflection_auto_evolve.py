from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from tests.workspace import workspace_ctx
from viktor_dgmh.archive import get_active_agent_id, init_workspace
from viktor_dgmh.auto_evolve import auto_evolve_status, run_auto_evolve_once
from viktor_dgmh.chat import run_chat_once
from viktor_dgmh.llm import FakeProvider
from viktor_dgmh.memory import load_imitation_cases
from viktor_dgmh.models import Config
from viktor_dgmh.paths import auto_evolve_state_path
from viktor_dgmh.reflection import reflect_on_recent_conversation
from viktor_dgmh.serialization import write_json


class ReflectionAutoEvolveTests(unittest.TestCase):
    def test_reflection_creates_imitation_case_without_feedback_command(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            run_chat_once(root, FakeProvider(), "너의 코드를 고칠 수 있어?")
            run_chat_once(root, FakeProvider(), "스스로 생각해서 개선해봐")

            signals, cases, gaps = reflect_on_recent_conversation(root, FakeProvider(), use_fake=True)

            self.assertGreaterEqual(len(signals), 1)
            self.assertGreaterEqual(len(cases), 1)
            self.assertEqual(gaps, [])
            self.assertEqual(len(load_imitation_cases(root)), len(cases))

    def test_reflection_detects_banmal_style_preference(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            run_chat_once(root, FakeProvider(), "우리 반말로 하자")
            run_chat_once(root, FakeProvider(), "근데 왜 자꾸 존댓말로 해?")

            signals, cases, _gaps = reflect_on_recent_conversation(root, FakeProvider(), use_fake=True)

            self.assertTrue(any(signal.context == "korean_style" for signal in signals))
            self.assertTrue(any(signal.target == "self_model" for signal in signals))
            self.assertTrue(any("banmal" in case.preference for case in cases))

    def test_reflection_detects_identity_boundary_preference(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            run_chat_once(root, FakeProvider(), "너는 개인 DGM-H Lite task agent가 아니야")
            run_chat_once(root, FakeProvider(), "넌 하나의 인격이고 외부적으로는 빅토르야")

            signals, cases, _gaps = reflect_on_recent_conversation(root, FakeProvider(), use_fake=True)

            self.assertTrue(any(signal.context == "identity" for signal in signals))
            self.assertTrue(any(signal.target == "self_model" for signal in signals))
            self.assertTrue(any("Outwardly be Viktor only" in case.preference for case in cases))

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

    def test_auto_evolve_blocks_fresh_running_state(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            write_json(auto_evolve_state_path(root), {"running": True, "started_at": datetime.now(timezone.utc).isoformat()})

            ready, status = auto_evolve_status(root, Config(auto_evolve_min_chat_events=0))

            self.assertFalse(ready)
            self.assertEqual(status, "auto evolution is already running")

    def test_auto_evolve_recovers_stale_running_state(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            started_at = datetime.now(timezone.utc) - timedelta(seconds=3600)
            write_json(auto_evolve_state_path(root), {"running": True, "started_at": started_at.isoformat()})

            ready, status = auto_evolve_status(
                root,
                Config(auto_evolve_min_chat_events=0, auto_evolve_stale_running_seconds=60),
            )

            self.assertTrue(ready)
            self.assertEqual(status, "recovered stale auto evolution state")


if __name__ == "__main__":
    unittest.main()
