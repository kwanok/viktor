from __future__ import annotations

import unittest

from viktor_dgmh.slack_app import REACTION_TO_FEEDBACK, _strip_bot_mentions, should_respond_to_channel_message


class SlackAppTests(unittest.TestCase):
    def test_reaction_mapping_matches_feedback_commands(self) -> None:
        self.assertEqual(REACTION_TO_FEEDBACK["scissors"], "/too-long")
        self.assertEqual(REACTION_TO_FEEDBACK["mag"], "/weak-evidence")
        self.assertEqual(REACTION_TO_FEEDBACK["warning"], "/unsafe")

    def test_strip_bot_mentions(self) -> None:
        self.assertEqual(_strip_bot_mentions("<@U123> evaluator 설명해줘").strip(), "evaluator 설명해줘")

    def test_channel_router_responds_to_judgment_question(self) -> None:
        decision = should_respond_to_channel_message("이 LangGraph 설계 괜찮을까?", min_score=0.65)

        self.assertTrue(decision.should_respond)
        self.assertGreaterEqual(decision.score, 0.65)

    def test_channel_router_stays_silent_for_chatter(self) -> None:
        decision = should_respond_to_channel_message("ㅇㅋ ㅋㅋ", min_score=0.65)

        self.assertFalse(decision.should_respond)


if __name__ == "__main__":
    unittest.main()
