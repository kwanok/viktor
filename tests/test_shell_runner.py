from __future__ import annotations

import os
import shutil
import unittest

from viktor_dgmh.models import Config
from viktor_dgmh.paths import shell_commands_path
from viktor_dgmh.serialization import read_jsonl
from viktor_dgmh.shell_runner import (
    format_shell_result,
    parse_shell_command,
    run_shell_command,
    shell_enabled,
    shell_user_allowed,
)

from tests.workspace import workspace_ctx


class ShellRunnerTests(unittest.TestCase):
    def test_parse_shell_command(self) -> None:
        self.assertEqual(parse_shell_command("!sh pwd"), "pwd")
        self.assertEqual(parse_shell_command(" !bash printf hi "), "printf hi")
        self.assertIsNone(parse_shell_command("pwd"))
        self.assertIsNone(parse_shell_command("!sh   "))

    def test_shell_enabled_defaults_to_config_and_env_can_override(self) -> None:
        old_value = os.environ.pop("SLACK_ENABLE_SHELL", None)
        try:
            self.assertFalse(shell_enabled(Config()))
            self.assertTrue(shell_enabled(Config(slack_shell_enabled=True)))
            os.environ["SLACK_ENABLE_SHELL"] = "true"
            self.assertTrue(shell_enabled(Config()))
            os.environ["SLACK_ENABLE_SHELL"] = "false"
            self.assertFalse(shell_enabled(Config(slack_shell_enabled=True)))
        finally:
            if old_value is None:
                os.environ.pop("SLACK_ENABLE_SHELL", None)
            else:
                os.environ["SLACK_ENABLE_SHELL"] = old_value

    def test_shell_user_allowlist(self) -> None:
        old_allow_any = os.environ.pop("SLACK_SHELL_ALLOW_ANY", None)
        old_allowed = os.environ.pop("SLACK_SHELL_ALLOWED_USERS", None)
        try:
            self.assertFalse(shell_user_allowed("U123"))
            os.environ["SLACK_SHELL_ALLOWED_USERS"] = "U123,U456"
            self.assertTrue(shell_user_allowed("U123"))
            self.assertFalse(shell_user_allowed("U999"))
            os.environ["SLACK_SHELL_ALLOW_ANY"] = "true"
            self.assertTrue(shell_user_allowed("U999"))
        finally:
            if old_allow_any is None:
                os.environ.pop("SLACK_SHELL_ALLOW_ANY", None)
            else:
                os.environ["SLACK_SHELL_ALLOW_ANY"] = old_allow_any
            if old_allowed is None:
                os.environ.pop("SLACK_SHELL_ALLOWED_USERS", None)
            else:
                os.environ["SLACK_SHELL_ALLOWED_USERS"] = old_allowed

    @unittest.skipIf(shutil.which("bash") is None, "bash is not available")
    def test_run_shell_command_logs_result(self) -> None:
        with workspace_ctx() as root:
            record = run_shell_command(
                root,
                "printf hello",
                config=Config(slack_shell_max_output_chars=100),
                user="U123",
                channel="C123",
                slack_ts="1.23",
            )

            self.assertEqual(record.exit_code, 0)
            self.assertEqual(record.stdout, "hello")
            rows = read_jsonl(shell_commands_path(root))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["command"], "printf hello")

    @unittest.skipIf(shutil.which("bash") is None, "bash is not available")
    def test_format_shell_result_escapes_code_fences(self) -> None:
        with workspace_ctx() as root:
            record = run_shell_command(
                root,
                "printf '```'",
                config=Config(slack_shell_max_output_chars=100),
            )

            result = format_shell_result(record)
            self.assertIn("` ` `", result)


if __name__ == "__main__":
    unittest.main()
