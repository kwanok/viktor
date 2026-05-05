from __future__ import annotations

import subprocess
import sys
import unittest
import os
from pathlib import Path

from tests.workspace import workspace_ctx


class CliTests(unittest.TestCase):
    def test_cli_init_and_inspect(self) -> None:
        with workspace_ctx() as root:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path.cwd())
            init_result = subprocess.run(
                [sys.executable, "-m", "viktor_dgmh", "init"],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(init_result.returncode, 0, init_result.stderr)
            inspect_result = subprocess.run(
                [sys.executable, "-m", "viktor_dgmh", "inspect", "--agent", "active"],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(inspect_result.returncode, 0, inspect_result.stderr)
            self.assertIn("Hyperagent: gen000_seed", inspect_result.stdout)
            self.assertIn("Self model: name=Viktor", inspect_result.stdout)

    def test_cli_chat_once_records_feedback(self) -> None:
        with workspace_ctx() as root:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path.cwd())
            init_result = subprocess.run(
                [sys.executable, "-m", "viktor_dgmh", "init"],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(init_result.returncode, 0, init_result.stderr)
            chat_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "viktor_dgmh",
                    "chat",
                    "--fake",
                    "--prompt",
                    "evaluator가 뭐야?",
                    "--feedback",
                    "/good",
                ],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(chat_result.returncode, 0, chat_result.stderr)
            self.assertTrue((root / "memory" / "preferences.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
