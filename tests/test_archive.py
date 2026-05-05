from __future__ import annotations

import unittest

from viktor_dgmh.archive import get_active_agent_id, init_workspace, load_agent, set_active_agent
from viktor_dgmh.defaults import DEFAULT_JUDGE_POLICY, DEFAULT_MUTATOR_STRATEGY, DEFAULT_REFLECTION_POLICY, DEFAULT_SELF_MODEL, SEED_TASK_PROMPT
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
            self.assertTrue((record.path / "self_model.yaml").exists())
            self.assertTrue((record.path / "reflection_policy.yaml").exists())
            self.assertTrue((record.path / "mutator_strategy.yaml").exists())
            self.assertTrue((record.path / "judge_policy.yaml").exists())

    def test_manual_active_update_validates_agent_exists(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            set_active_agent(root, "gen000_seed")
            self.assertEqual(select_parent(root), "gen000_seed")

    def test_seed_prompt_prefers_concise_feedback_aware_self_improvement(self) -> None:
        self.assertIn("one short sentence first", SEED_TASK_PROMPT)
        self.assertIn("direct user feedback", SEED_TASK_PROMPT)
        self.assertIn("improve yourself", SEED_TASK_PROMPT)

    def test_seed_self_model_keeps_internal_identity_private(self) -> None:
        first_line = SEED_TASK_PROMPT.splitlines()[0]
        self.assertEqual(first_line, "Behavior model for Viktor.")
        self.assertEqual(DEFAULT_SELF_MODEL["name"], "Viktor")
        self.assertEqual(DEFAULT_SELF_MODEL["default_tone"], "banmal")
        self.assertIn("assistant", DEFAULT_SELF_MODEL["forbidden_self_descriptions"])
        self.assertIn("DGM-H", DEFAULT_SELF_MODEL["internal_only"])

    def test_seed_meta_strategy_defaults_are_present(self) -> None:
        self.assertIn("identity", DEFAULT_REFLECTION_POLICY["self_model_contexts"])
        self.assertIn("reflection_policy.yaml", DEFAULT_MUTATOR_STRATEGY["editable_files"])
        self.assertIn("quality of future self-improvement strategy", DEFAULT_JUDGE_POLICY["evaluation_axes"])


if __name__ == "__main__":
    unittest.main()
