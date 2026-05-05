from __future__ import annotations

import unittest

from viktor_dgmh.archive import init_workspace
from viktor_dgmh.validator import validate_agent_dir
from tests.workspace import workspace_ctx


class ValidatorTests(unittest.TestCase):
    def test_allows_pure_helper(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            result = validate_agent_dir(root / "archive" / "gen000_seed")
            self.assertTrue(result.passed, result.issues)

    def test_rejects_open_call(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            helper = root / "archive" / "gen000_seed" / "helpers.py"
            helper.write_text("def bad():\n    return open('x').read()\n", encoding="utf-8")
            result = validate_agent_dir(root / "archive" / "gen000_seed")
            self.assertFalse(result.passed)
            self.assertTrue(any("open" in issue.message for issue in result.issues))

    def test_rejects_subprocess_import(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            helper = root / "archive" / "gen000_seed" / "helpers.py"
            helper.write_text("import subprocess\n\ndef bad():\n    return 1\n", encoding="utf-8")
            result = validate_agent_dir(root / "archive" / "gen000_seed")
            self.assertFalse(result.passed)
            self.assertTrue(any("subprocess" in issue.message for issue in result.issues))


if __name__ == "__main__":
    unittest.main()
