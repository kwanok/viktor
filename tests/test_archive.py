from __future__ import annotations

import unittest

from viktor_dgmh.archive import get_active_agent_id, init_workspace, load_agent, set_active_agent
from viktor_dgmh.selection import select_parent
from tests.workspace import workspace_ctx


class ArchiveTests(unittest.TestCase):
    def test_init_creates_seed_and_active_pointer(self) -> None:
        with workspace_ctx() as root:
            seed = init_workspace(root)
            self.assertEqual(seed, "gen000_seed")
            self.assertEqual(get_active_agent_id(root), "gen000_seed")
            record = load_agent(root, seed)
            self.assertEqual(record.manifest.generation, 0)
            self.assertGreaterEqual(record.scores.safety, 0.9)

    def test_manual_active_update_validates_agent_exists(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            set_active_agent(root, "gen000_seed")
            self.assertEqual(select_parent(root), "gen000_seed")


if __name__ == "__main__":
    unittest.main()
