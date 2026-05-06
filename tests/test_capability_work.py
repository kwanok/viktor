from __future__ import annotations

import unittest

from tests.workspace import workspace_ctx
from viktor_dgmh.archive import init_workspace
from viktor_dgmh.capability_work import approve_work_from_message, execute_capability_work, request_capability_work
from viktor_dgmh.memory import append_capability_gap, load_capability_work_items, load_latest_capability_work_items
from viktor_dgmh.models import CapabilityGap, Config
from viktor_dgmh.runtime_lifecycle import runtime_status


class CapabilityWorkTests(unittest.TestCase):
    def test_request_capability_work_blocks_external_scope_change(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            gap = CapabilityGap(
                gap_id="gap_scope",
                source="test",
                summary="Need Slack reaction capability.",
                requested_capability="slack_reaction_add",
                failure_mode="missing_slack_action_capability",
                required_changes=["slack_scope", "code", "restart"],
                requires_restart=True,
            )
            append_capability_gap(root, gap)

            item = request_capability_work(root, gap, source="test", requested_by="U1")

            self.assertEqual(item.status, "blocked")
            self.assertIn("external approval", item.blocked_reason or "")
            self.assertTrue(item.requires_restart)

    def test_request_capability_work_deduplicates_active_item(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            gap = CapabilityGap(
                gap_id="gap_memory",
                source="test",
                summary="Need better retrieval behavior.",
                requested_capability="memory_retrieval",
                failure_mode="missing_memory_behavior",
                required_changes=["code", "tests"],
            )

            first = request_capability_work(root, gap, source="test")
            second = request_capability_work(root, gap, source="test")

            self.assertEqual(first.work_id, second.work_id)
            self.assertEqual(first.status, "queued")
            self.assertEqual(len(load_capability_work_items(root)), 1)

    def test_approval_message_queues_blocked_work(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            gap = CapabilityGap(
                gap_id="gap_scope",
                source="test",
                summary="Need Slack reaction capability.",
                requested_capability="slack_reaction_add",
                failure_mode="missing_slack_action_capability",
                required_changes=["slack_scope", "code", "restart"],
                requires_restart=True,
            )
            append_capability_gap(root, gap)
            blocked = request_capability_work(root, gap, source="test", requested_by="U1")

            queued = approve_work_from_message(
                root,
                "reactions:write 권한 추가했고 직접 작업해서 결과 알려줘",
                "user:U1: Please add :eyes: as a reaction.",
                approved_by="U1",
            )

            self.assertIsNotNone(queued)
            self.assertEqual(queued.work_id, blocked.work_id)
            self.assertEqual(queued.status, "queued")
            self.assertIsNone(queued.blocked_reason)
            self.assertEqual(load_latest_capability_work_items(root)[0].status, "queued")

    def test_fake_execute_completes_and_requests_restart(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            gap = CapabilityGap(
                gap_id="gap_restart",
                source="test",
                summary="Need code change and restart.",
                requested_capability="runtime_change",
                failure_mode="missing_runtime_behavior",
                required_changes=["code", "restart"],
                requires_restart=True,
            )
            item = request_capability_work(root, gap, source="test")

            completed = execute_capability_work(root, item.work_id, Config(), fake=True)
            status = runtime_status(root)

            self.assertEqual(completed.status, "completed")
            self.assertIsNotNone(completed.restart_request_id)
            self.assertEqual(status["restart_request"]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
