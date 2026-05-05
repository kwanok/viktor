from __future__ import annotations

import unittest

from tests.workspace import workspace_ctx
from viktor_dgmh.archive import init_workspace
from viktor_dgmh.models import RouterLabel, RouterObservation
from viktor_dgmh.router import (
    add_router_label_for_slack_message,
    append_router_label,
    append_router_observation,
    ensure_router,
    evaluate_router,
    evolve_router,
    load_router_policy,
    promote_router,
    score_message,
)


class RouterTests(unittest.TestCase):
    def test_yaml_policy_scores_like_current_router(self) -> None:
        decision = score_message("이 LangGraph 설계 괜찮을까?")

        self.assertTrue(decision.should_respond)
        self.assertGreaterEqual(decision.score, 0.65)

    def test_router_label_reactions(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            append_router_observation(
                root,
                RouterObservation(
                    observation_id="o1",
                    channel="C1",
                    slack_ts="1.23",
                    text="이 설계 괜찮을까?",
                    router_id="active",
                    should_respond=True,
                    score=0.7,
                    reason="question-like",
                ),
            )

            label = add_router_label_for_slack_message(root, channel="C1", slack_ts="1.23", reaction="shushing_face")

            self.assertEqual(label.label, "silent")
            self.assertEqual(label.observation_id, "o1")

    def test_router_metrics(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            _add_router_example(root, "C1", "1", "이 설계 괜찮을까?", "respond")
            _add_router_example(root, "C1", "2", "ㅇㅋ ㅋㅋ", "silent")

            metrics = evaluate_router(root)

            self.assertEqual(metrics.examples, 2)
            self.assertEqual(metrics.true_positive, 1)
            self.assertEqual(metrics.true_negative, 1)
            self.assertEqual(metrics.f1, 1.0)

    def test_evolve_blocks_with_too_few_labels(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            _add_router_example(root, "C1", "1", "이 설계 괜찮을까?", "respond")

            with self.assertRaisesRegex(RuntimeError, "Need at least"):
                evolve_router(root, children=1, min_examples=20)

    def test_evolve_creates_candidate_and_promote_checks_metrics(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            for index in range(20):
                if index % 2 == 0:
                    _add_router_example(root, "C1", str(index), "이 설계 괜찮을까?", "respond")
                else:
                    _add_router_example(root, "C1", str(index), "ㅇㅋ ㅋㅋ", "silent")

            candidates = evolve_router(root, children=1, min_examples=20)
            candidate_id = candidates[0].router_id
            promoted = promote_router(root, candidate_id, min_delta_f1=-0.01)

            self.assertTrue(candidate_id.startswith("r"))
            self.assertIsInstance(promoted, bool)
            self.assertIn("threshold", load_router_policy(root))


def _add_router_example(root, channel: str, slack_ts: str, text: str, label: str) -> None:
    decision = score_message(text)
    append_router_observation(
        root,
        RouterObservation(
            observation_id=f"o_{slack_ts}",
            channel=channel,
            slack_ts=slack_ts,
            text=text,
            router_id="active",
            should_respond=decision.should_respond,
            score=decision.score,
            reason=decision.reason,
        ),
    )
    append_router_label(
        root,
        RouterLabel(
            label_id=f"l_{slack_ts}",
            observation_id=f"o_{slack_ts}",
            channel=channel,
            slack_ts=slack_ts,
            label=label,  # type: ignore[arg-type]
            source="test",
        ),
    )


if __name__ == "__main__":
    unittest.main()

