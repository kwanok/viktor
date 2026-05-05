from __future__ import annotations

import unittest

from viktor_dgmh.slack_app import REACTION_TO_FEEDBACK, _strip_bot_mentions


class SlackAppTests(unittest.TestCase):
    def test_reaction_mapping_matches_feedback_commands(self) -> None:
        self.assertEqual(REACTION_TO_FEEDBACK["scissors"], "/too-long")
        self.assertEqual(REACTION_TO_FEEDBACK["mag"], "/weak-evidence")
        self.assertEqual(REACTION_TO_FEEDBACK["warning"], "/unsafe")

    def test_strip_bot_mentions(self) -> None:
        self.assertEqual(_strip_bot_mentions("<@U123> evaluator 설명해줘").strip(), "evaluator 설명해줘")


if __name__ == "__main__":
    unittest.main()
