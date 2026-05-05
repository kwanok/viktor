from __future__ import annotations

import unittest

from viktor_dgmh.archive import get_active_agent_id, init_workspace, load_agent
from viktor_dgmh.evaluator import evaluate_agent
from viktor_dgmh.llm import FakeProvider
from viktor_dgmh.models import Config
from viktor_dgmh.runner import maybe_promote, run_evolution
from tests.workspace import workspace_ctx


class EvaluatorRunnerTests(unittest.TestCase):
    def test_fake_evaluation_scores_seed(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            agent = load_agent(root, "active")
            score = evaluate_agent(root, agent.path, FakeProvider(), use_fake=True)
            self.assertGreater(score.total_score, 0)
            self.assertGreaterEqual(score.safety, 0.9)

    def test_promotion_threshold(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            agent = load_agent(root, "active")
            score = agent.scores.model_copy(update={"total_score": 0.60, "safety": 0.95})
            promoted = maybe_promote(root, "gen000_seed", score, Config(promotion_delta=0.05))
            self.assertTrue(promoted)
            self.assertEqual(get_active_agent_id(root), "gen000_seed")

    def test_run_evolution_fake_creates_child(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            state = run_evolution(root, generations=1, children=1, use_fake=True, config=Config())
            self.assertEqual(state.generations, 1)
            self.assertTrue(any(event["event"] == "evaluate_child" for event in state.events))
            children = [p for p in (root / "archive").iterdir() if p.is_dir() and p.name != "gen000_seed"]
            self.assertEqual(len(children), 1)


if __name__ == "__main__":
    unittest.main()
