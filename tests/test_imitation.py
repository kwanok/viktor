from __future__ import annotations

import json
import unittest

from tests.workspace import workspace_ctx
from viktor_dgmh.archive import init_workspace
from viktor_dgmh.chat import run_chat_once
from viktor_dgmh.imitation import aggregate_pairwise_results, evaluate_pairwise_imitation, should_promote_pairwise
from viktor_dgmh.llm import FakeProvider
from viktor_dgmh.memory import append_preference
from viktor_dgmh.models import ImitationCase, PairwiseResult, PreferenceSignal
from viktor_dgmh.mutator import create_child


class CapturingMutationProvider:
    def __init__(self) -> None:
        self.messages = []

    def chat(self, messages, *, model=None, response_format=None) -> str:
        self.messages = messages
        files = json.loads(messages[-1]["content"].split("Current files:\n", 1)[1])
        files["reflection_policy.yaml"] = files["reflection_policy.yaml"].replace(
            "- tone\n",
            "- tone\n- identity_boundary\n",
        )
        files["self_model.yaml"] = files["self_model.yaml"].replace(
            "public_identity_rules:\n",
            "public_identity_rules:\n- Use casual Korean banmal with the user by default.\n",
        )
        return json.dumps({"mutation_summary": "Encode banmal preference.", "files": files}, ensure_ascii=False)


class ImitationTests(unittest.TestCase):
    def test_pairwise_aggregate_candidate_win_rate(self) -> None:
        cases = [
            ImitationCase(id="c1", source_signal_id="s1", prompt="p", preference="direct", weight=1.0),
            ImitationCase(id="c2", source_signal_id="s2", prompt="p", preference="evidence", weight=2.0),
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
            run_chat_once(root, FakeProvider(), "설계 판단은 어떻게 해야 해?", feedback_text="/too-long")
            child_id, _ = create_child(root, "gen000_seed", 1, 1, FakeProvider(), use_fake=True)

            aggregate = evaluate_pairwise_imitation(root, child_id, FakeProvider(), use_fake=True)

            self.assertIsNotNone(aggregate)
            self.assertGreaterEqual(aggregate.weighted_win_rate, 0.60)
            self.assertEqual(aggregate.safety_regressions, 0)

    def test_mutator_passes_memory_brief_to_meta_agent(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            append_preference(
                root,
                PreferenceSignal(
                    signal_id="s1",
                    session_id="chat1",
                    source_event_id="p1",
                    kind="reflection_tone",
                    polarity="negative",
                    strength=0.9,
                    context="korean_style",
                    target="self_model",
                    text="Use casual Korean banmal with the user by default.",
                ),
            )
            provider = CapturingMutationProvider()

            _child_id, child_path = create_child(root, "gen000_seed", 1, 1, provider)

            mutation_prompt = provider.messages[-1]["content"]
            self.assertIn("Evolution brief", mutation_prompt)
            self.assertIn("banmal", mutation_prompt)
            self.assertIn('"target": "self_model"', mutation_prompt)
            self.assertIn("self_model.yaml", mutation_prompt)
            self.assertIn("Active mutator strategy", mutation_prompt)
            self.assertIn("reflection_policy.yaml", mutation_prompt)
            self.assertIn("banmal", (child_path / "self_model.yaml").read_text(encoding="utf-8"))
            self.assertIn("identity_boundary", (child_path / "reflection_policy.yaml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
