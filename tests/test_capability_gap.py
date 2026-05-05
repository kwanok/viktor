from __future__ import annotations

import unittest

from tests.workspace import workspace_ctx
from viktor_dgmh.archive import init_workspace
from viktor_dgmh.chat import run_chat_once
from viktor_dgmh.llm import FakeProvider
from viktor_dgmh.memory import append_capability_gap, load_capability_gaps
from viktor_dgmh.meta_agent import MetaAgent
from viktor_dgmh.models import CapabilityGap
from viktor_dgmh.reflection import reflect_on_recent_conversation


class CapabilityGapTests(unittest.TestCase):
    def test_append_and_load_capability_gap(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            append_capability_gap(
                root,
                CapabilityGap(
                    gap_id="gap_test",
                    source="test",
                    summary="Missing Slack reaction action.",
                    evidence=["User asked for a reaction."],
                    requested_capability="slack_reaction_add",
                    failure_mode="missing_slack_action_capability",
                    required_changes=["slack_scope", "code", "restart"],
                    requires_restart=True,
                ),
            )

            gaps = load_capability_gaps(root)

            self.assertEqual(len(gaps), 1)
            self.assertEqual(gaps[0].requested_capability, "slack_reaction_add")
            self.assertTrue(gaps[0].requires_restart)

    def test_fake_reflection_records_eyes_reaction_as_capability_gap(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            run_chat_once(root, FakeProvider(), "앞으로 대답하기 전에 :eyes: 이거 달아줄 수 있어?")
            run_chat_once(root, FakeProvider(), "아니 내 메시지에 이모지로 달아주라고. 해봐")

            _signals, _cases, gaps = reflect_on_recent_conversation(root, FakeProvider(), use_fake=True)

            self.assertEqual(len(gaps), 1)
            gap = gaps[0]
            self.assertEqual(gap.requested_capability, "slack_reaction_add")
            self.assertEqual(gap.required_changes, ["slack_scope", "code", "restart"])
            self.assertTrue(gap.requires_restart)

    def test_meta_agent_evolution_brief_includes_capability_gaps(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            append_capability_gap(
                root,
                CapabilityGap(
                    gap_id="gap_test",
                    source="test",
                    summary="Need a supervised restart path.",
                    evidence=["Code changes are not reflected in a running Slack process."],
                    requested_capability="runtime_supervised_restart",
                    failure_mode="runtime_lifecycle_gap",
                    required_changes=["supervisor", "restart"],
                    requires_restart=False,
                ),
            )

            brief = MetaAgent(root).build_evolution_brief()

            self.assertIn("recent_capability_gaps", brief)
            self.assertEqual(brief["recent_capability_gaps"][0]["requested_capability"], "runtime_supervised_restart")


if __name__ == "__main__":
    unittest.main()
