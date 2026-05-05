from __future__ import annotations

import subprocess
import sys
import unittest
import os
from pathlib import Path

from tests.workspace import workspace_ctx
from viktor_dgmh.memory import append_capability_gap
from viktor_dgmh.models import CapabilityGap


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

    def test_cli_capability_list_and_inspect(self) -> None:
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
            append_capability_gap(
                root,
                CapabilityGap(
                    gap_id="gap_cli",
                    source="test",
                    summary="Need supervised restart.",
                    requested_capability="runtime_supervised_restart",
                    failure_mode="runtime_lifecycle_gap",
                ),
            )

            list_result = subprocess.run(
                [sys.executable, "-m", "viktor_dgmh", "capability", "list"],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            inspect_result = subprocess.run(
                [sys.executable, "-m", "viktor_dgmh", "capability", "inspect", "--gap", "gap_cli"],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(list_result.returncode, 0, list_result.stderr)
            self.assertIn("gap_cli", list_result.stdout)
            self.assertEqual(inspect_result.returncode, 0, inspect_result.stderr)
            self.assertIn("runtime_supervised_restart", inspect_result.stdout)

    def test_cli_capability_work_and_work_list(self) -> None:
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
            append_capability_gap(
                root,
                CapabilityGap(
                    gap_id="gap_work_cli",
                    source="test",
                    summary="Need Slack reaction capability.",
                    requested_capability="slack_reaction_add",
                    failure_mode="missing_slack_action_capability",
                    required_changes=["slack_scope", "code", "restart"],
                    requires_restart=True,
                ),
            )

            work_result = subprocess.run(
                [sys.executable, "-m", "viktor_dgmh", "capability", "work", "--gap", "gap_work_cli"],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            list_result = subprocess.run(
                [sys.executable, "-m", "viktor_dgmh", "capability", "work-list"],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(work_result.returncode, 0, work_result.stderr)
            self.assertIn("Status: blocked", work_result.stdout)
            self.assertEqual(list_result.returncode, 0, list_result.stderr)
            self.assertIn("gap_work_cli", list_result.stdout)

    def test_cli_runtime_request_restart_and_status(self) -> None:
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

            request_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "viktor_dgmh",
                    "runtime",
                    "request-restart",
                    "--reason",
                    "test restart",
                ],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            status_result = subprocess.run(
                [sys.executable, "-m", "viktor_dgmh", "runtime", "status"],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(request_result.returncode, 0, request_result.stderr)
            self.assertIn("Restart requested:", request_result.stdout)
            self.assertEqual(status_result.returncode, 0, status_result.stderr)
            self.assertIn("Restart request: pending", status_result.stdout)


if __name__ == "__main__":
    unittest.main()
