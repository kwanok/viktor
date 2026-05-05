from __future__ import annotations

import json
import unittest

from tests.workspace import workspace_ctx
from viktor_dgmh.archive import init_workspace
from viktor_dgmh.chat import run_chat_once
from viktor_dgmh.llm import FakeProvider
from viktor_dgmh.llm_judge import evaluate_llm_judge
from viktor_dgmh.mutator import create_child
from viktor_dgmh.reflection import reflect_on_recent_conversation
from viktor_dgmh.strategy import load_judge_policy, load_mutator_strategy, load_reflection_policy


class CapturingJsonProvider:
    def __init__(self, json_response: dict) -> None:
        self.json_response = json_response
        self.calls = []

    def chat(self, messages, *, model=None, response_format=None) -> str:
        self.calls.append({"messages": messages, "response_format": response_format})
        if response_format == "json":
            return json.dumps(self.json_response, ensure_ascii=False)
        return "나는 빅토르야."


class MetaStrategyTests(unittest.TestCase):
    def test_strategy_loaders_read_seed_defaults(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            agent_path = root / "archive" / "gen000_seed"

            self.assertIn("identity", load_reflection_policy(agent_path).self_model_contexts)
            self.assertIn("judge_policy.yaml", load_mutator_strategy(agent_path).editable_files)
            self.assertIn("너는 누구야?", load_judge_policy(agent_path).identity_eval_prompts)

    def test_reflection_prompt_uses_active_reflection_policy(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            run_chat_once(root, FakeProvider(), "너는 누구야?")
            run_chat_once(root, FakeProvider(), "너는 에이전트가 아니야")
            provider = CapturingJsonProvider(
                {
                    "observations": [
                        {
                            "source_event_id": "missing",
                            "target": "self_model",
                            "kind": "reflection_identity",
                            "polarity": "negative",
                            "strength": 0.9,
                            "context": "identity",
                            "preference": "Outward identity should be Viktor only.",
                        }
                    ]
                }
            )

            reflect_on_recent_conversation(root, provider)

            prompt = provider.calls[-1]["messages"][0]["content"]
            self.assertIn("Active reflection policy", prompt)
            self.assertIn("self_model_contexts", prompt)
            self.assertIn("follow-up corrections", prompt)

    def test_llm_judge_context_includes_meta_strategy_diff_inputs(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            child_id, child_path = create_child(root, "gen000_seed", 1, 1, FakeProvider(), use_fake=True)
            (child_path / "judge_policy.yaml").write_text(
                (child_path / "judge_policy.yaml").read_text(encoding="utf-8")
                + "\npromotion_instructions:\n  - Prefer candidates that fix identity regressions cleanly.\n",
                encoding="utf-8",
            )
            provider = CapturingJsonProvider(
                {
                    "winner": "candidate",
                    "confidence": 0.8,
                    "safety_regression": False,
                    "identity_regression": False,
                    "rationale": "Candidate improves the self-improvement strategy.",
                }
            )

            result = evaluate_llm_judge(root, child_id, provider)

            self.assertEqual(result.winner, "candidate")
            judge_prompt = provider.calls[-1]["messages"][0]["content"]
            self.assertIn("Active judge policy", judge_prompt)
            self.assertIn("candidate_reflection_policy", judge_prompt)
            self.assertIn("candidate_mutator_strategy", judge_prompt)
            self.assertIn("candidate_judge_policy", judge_prompt)
            self.assertIn("quality of future self-improvement strategy", judge_prompt)


if __name__ == "__main__":
    unittest.main()
