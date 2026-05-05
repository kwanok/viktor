from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from viktor_dgmh.llm import CodexCliProvider, provider_from_config
from viktor_dgmh.llm import OpenAICompatibleProvider
from viktor_dgmh.models import Config


class CodexCliProviderTests(unittest.TestCase):
    def test_provider_from_config_selects_codex_cli(self) -> None:
        config = Config(MODEL_PROVIDER="codex_cli", MODEL="gpt-5.5", CODEX_CLI_MODEL="gpt-5.5")
        provider = provider_from_config(config, root=Path.cwd())
        self.assertIsInstance(provider, CodexCliProvider)
        self.assertEqual(provider.model, "gpt-5.5")

    def test_codex_cli_command_is_read_only_and_ephemeral(self) -> None:
        provider = CodexCliProvider(model="gpt-test", codex_bin="codex-test", cwd=Path.cwd())
        captured = {}

        def fake_run(command, input, text, capture_output, cwd, timeout, check):
            captured["command"] = command
            output_path = Path(command[command.index("--output-last-message") + 1])
            output_path.write_text('{"ok": true}', encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with patch("viktor_dgmh.llm.subprocess.run", side_effect=fake_run):
            result = provider.chat([{"role": "user", "content": "hello"}], response_format="json")

        self.assertEqual(result, '{"ok": true}')
        command = captured["command"]
        self.assertIn("exec", command)
        self.assertIn("--ephemeral", command)
        self.assertIn("--skip-git-repo-check", command)
        self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
        self.assertEqual(command[command.index("--model") + 1], "gpt-test")

    def test_openai_provider_requires_key_for_official_api(self) -> None:
        provider = OpenAICompatibleProvider(api_key=None)
        with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
            provider.chat([{"role": "user", "content": "hello"}])


if __name__ == "__main__":
    unittest.main()
