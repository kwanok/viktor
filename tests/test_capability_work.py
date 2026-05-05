from __future__ import annotations

import unittest

from tests.workspace import workspace_ctx
from viktor_dgmh.archive import init_workspace
from viktor_dgmh.capability_work import request_capability_work
from viktor_dgmh.memory import append_capability_gap, load_capability_work_items
from viktor_dgmh.models import CapabilityGap


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


if __name__ == "__main__":
    unittest.main()
