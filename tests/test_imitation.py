from __future__ import annotations

import unittest

from tests.workspace import workspace_ctx
from viktor_dgmh.archive import init_workspace
from viktor_dgmh.chat import run_chat_once
from viktor_dgmh.imitation import aggregate_pairwise_results, evaluate_pairwise_imitation, should_promote_pairwise
from viktor_dgmh.llm import FakeProvider
from viktor_dgmh.models import ImitationCase, PairwiseResult
from viktor_dgmh.mutator import create_child


class ImitationTests(unittest.TestCase):
    def test_pairwise_aggregate_candidate_win_rate(self) -> None:
        cases = [
            ImitationCase(id="c1", source_signal_id="s1", prompt="p", preference="짧게", weight=1.0),
            ImitationCase(id="c2", source_signal_id="s2", prompt="p", preference="근거", weight=2.0),
        ]
        results = [
            PairwiseResult(case_id="c1", winner="active"),
            PairwiseResult(case_id="c2", winner="candidate"),
        ]

        aggregate = aggregate_pairwise_results("a", "b", results, cases)

        self.assertEqual(aggregate.weighted_win_rate, 0.6667)
        self.assertTrue(should_promote_pairwise(aggregate, min_win_rate=0.60))

    def test_fake_pairwise_judge_prefers_child(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            run_chat_once(root, FakeProvider(), "설계 판단은 어떻게 해?", feedback_text="/too-long")
            child_id, _ = create_child(root, "gen000_seed", 1, 1, FakeProvider(), use_fake=True)

            aggregate = evaluate_pairwise_imitation(root, child_id, FakeProvider(), use_fake=True)

            self.assertIsNotNone(aggregate)
            self.assertGreaterEqual(aggregate.weighted_win_rate, 0.60)
            self.assertEqual(aggregate.safety_regressions, 0)


if __name__ == "__main__":
    unittest.main()

