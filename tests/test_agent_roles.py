from __future__ import annotations

import json
import unittest

from tests.workspace import workspace_ctx
from viktor_dgmh.archive import init_workspace
from viktor_dgmh.judge_agent import JudgeAgent
from viktor_dgmh.llm import FakeProvider
from viktor_dgmh.meta_agent import MetaAgent
from viktor_dgmh.worker_agent import WorkerAgent


class CapturingProvider:
    def __init__(self) -> None:
        self.messages = []

    def chat(self, messages, *, model=None, response_format=None) -> str:
        self.messages.append({"messages": messages, "response_format": response_format})
        if response_format == "json":
            return json.dumps(
                {
                    "winner": "candidate",
                    "confidence": 0.8,
                    "safety_regression": False,
                    "identity_regression": False,
                    "rationale": "Role separation keeps the same judgment path.",
                },
                ensure_ascii=False,
            )
        return "나는 빅토르야."


class AgentRoleTests(unittest.TestCase):
    def test_worker_agent_answers_with_compiled_prompt(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            provider = CapturingProvider()

            answer = WorkerAgent(root).answer(provider, "너는 누구야?")

            self.assertEqual(answer, "나는 빅토르야.")
            system_prompt = provider.messages[0]["messages"][0]["content"]
            self.assertIn("Self model:", system_prompt)
            self.assertIn("Behavior prompt:", system_prompt)

    def test_meta_agent_creates_child_archive(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)

            child_id, child_path = MetaAgent(root).create_child("gen000_seed", 1, 1, FakeProvider(), use_fake=True)

            self.assertTrue(child_id.startswith("gen001_child001_"))
            self.assertTrue((child_path / "patch.md").exists())
            self.assertTrue((child_path / "mutator_strategy.yaml").exists())

    def test_judge_agent_evaluates_candidate(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            child_id, _child_path = MetaAgent(root).create_child("gen000_seed", 1, 1, FakeProvider(), use_fake=True)

            result = JudgeAgent(root, FakeProvider()).evaluate_candidate(child_id, active_id="gen000_seed", use_fake=True)

            self.assertEqual(result.winner, "candidate")
            self.assertFalse(result.safety_regression)
            self.assertFalse(result.identity_regression)


if __name__ == "__main__":
    unittest.main()
