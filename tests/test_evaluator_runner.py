from __future__ import annotations

import unittest

from tests.workspace import workspace_ctx
from viktor_dgmh.archive import get_active_agent_id, init_workspace, load_agent
from viktor_dgmh.chat import run_chat_once
from viktor_dgmh.evaluator import evaluate_agent
from viktor_dgmh.llm import FakeProvider
from viktor_dgmh.llm_judge import evaluate_llm_judge
from viktor_dgmh.models import Config
from viktor_dgmh.runner import maybe_promote, run_evolution


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
            promoted = maybe_promote(root, "gen000_seed", score, Config(promotion_mode="score", promotion_delta=0.05))
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

    def test_run_evolution_llm_judge_promotes_fake_style_improvement(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            run_chat_once(root, FakeProvider(), "설계 판단은 어떻게 해야 해?", feedback_text="/too-long")
            state = run_evolution(root, generations=1, children=1, use_fake=True, config=Config())
            active_id = get_active_agent_id(root)

            self.assertTrue(any(event["event"] == "maybe_promote" for event in state.events))
            self.assertNotEqual(active_id, "gen000_seed")
            self.assertTrue((load_agent(root, active_id).path / "llm_judge_scores.json").exists())

    def test_llm_judge_result_prefers_fake_candidate(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            state = run_evolution(
                root,
                generations=1,
                children=1,
                use_fake=True,
                config=Config(promotion_mode="score"),
            )
            candidate_id = state.candidate_id
            result = evaluate_llm_judge(root, candidate_id, FakeProvider(), active_id="gen000_seed", use_fake=True)

            self.assertEqual(result.winner, "candidate")
            self.assertGreaterEqual(result.confidence, 0.65)


if __name__ == "__main__":
    unittest.main()
