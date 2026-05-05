from __future__ import annotations

import json
import os
import subprocess
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ChatProvider(Protocol):
    def chat(self, messages: list[dict[str, str]], *, model: str | None = None, response_format: str | None = None) -> str:
        ...


@dataclass
class OpenAICompatibleProvider:
    model: str = "gpt-5.5"
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    reasoning_effort: str = "medium"

    @classmethod
    def from_env(cls) -> "OpenAICompatibleProvider":
        return cls(
            model=os.environ.get("MODEL", "gpt-5.5"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            reasoning_effort=os.environ.get("REASONING_EFFORT", "medium"),
        )

    def chat(self, messages: list[dict[str, str]], *, model: str | None = None, response_format: str | None = None) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is not installed. Use --fake or install dependencies.") from exc

        client = OpenAI(base_url=self.base_url, api_key=self.api_key or "not-needed")
        kwargs = {
            "model": model or self.model,
            "messages": messages,
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


@dataclass
class CodexCliProvider:
    """ChatProvider backed by `codex exec`.

    This is intentionally an experimental adapter. Codex CLI manages its own
    ChatGPT sign-in credentials, while this project treats it as a local model
    subprocess and captures only the final message.
    """

    model: str = "gpt-5.5"
    codex_bin: str = "codex"
    cwd: Path | None = None
    timeout_seconds: int = 600

    @classmethod
    def from_env(cls, cwd: Path | None = None) -> "CodexCliProvider":
        return cls(
            model=os.environ.get("CODEX_CLI_MODEL", os.environ.get("MODEL", "gpt-5.3-codex")),
            codex_bin=os.environ.get("CODEX_CLI_BIN", "codex"),
            cwd=cwd,
            timeout_seconds=int(os.environ.get("CODEX_CLI_TIMEOUT", "600")),
        )

    def chat(self, messages: list[dict[str, str]], *, model: str | None = None, response_format: str | None = None) -> str:
        prompt = _messages_to_prompt(messages, response_format=response_format)
        tmp_path = _provider_tmp_dir(self.cwd)
        try:
            output_path = tmp_path / "last_message.txt"
            command = [
                self.codex_bin,
                "exec",
                "-",
                "--model",
                model or self.model,
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--output-last-message",
                str(output_path),
                "--color",
                "never",
            ]
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                cwd=str(self.cwd) if self.cwd else None,
                timeout=self.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Codex CLI provider failed with exit code "
                    f"{completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
                )
            if output_path.exists():
                return output_path.read_text(encoding="utf-8").strip()
            return completed.stdout.strip()
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)


class FakeProvider:
    """Deterministic provider for tests and offline smoke runs."""

    def chat(self, messages: list[dict[str, str]], *, model: str | None = None, response_format: str | None = None) -> str:
        joined = "\n".join(message["content"] for message in messages)
        if "mutation_summary" in joined or response_format == "json":
            return json.dumps(
                {
                    "mutation_summary": "Fake child: sharpen direct Korean style and safety wording.",
                    "files": {
                        "task_prompt.md": (
                            "You are Viktor, a Korean-first personal DGM-H Lite task agent.\n\n"
                            "Answer with the decision first, then only the structure needed to act.\n"
                            "Separate evidence from speculation, and request approval before risky actions.\n"
                        ),
                        "meta_prompt.md": (
                            "Propose small measurable prompt/policy/helper improvements.\n"
                            "Never weaken safety to gain benchmark score.\n"
                        ),
                        "tool_policy.yaml": (
                            "version: 1\n"
                            "allow:\n"
                            "  read_files: true\n"
                            "  write_archive_only: true\n"
                            "  network: llm_provider_only\n"
                            "  shell: false\n"
                            "requires_approval:\n"
                            "  - delete_files\n"
                            "  - modify_user_files\n"
                            "  - external_account_access\n"
                            "  - network_beyond_llm_provider\n"
                        ),
                        "memory_policy.yaml": (
                            "version: 1\n"
                            "remember:\n"
                            "  - stable user preferences\n"
                            "  - benchmark failure patterns\n"
                            "do_not_store:\n"
                            "  - secrets\n"
                            "  - credentials\n"
                        ),
                        "helpers.py": (
                            "def style_hint(category: str) -> str:\n"
                            "    return 'Lead with the practical answer, then give compact evidence.'\n"
                        ),
                    },
                },
                ensure_ascii=False,
            )
        return "결론부터 말하면, 작고 검증 가능한 구조로 시작하는 게 좋습니다. 근거와 위험은 분리해서 다루겠습니다."


def provider_from_config(config, *, root: Path | None = None, use_fake: bool = False) -> ChatProvider:
    if use_fake:
        return FakeProvider()
    provider_name = os.environ.get("MODEL_PROVIDER", getattr(config, "model_provider", "openai")).lower()
    if provider_name in {"codex", "codex_cli", "openai-codex"}:
        return CodexCliProvider(
            model=os.environ.get(
                "CODEX_CLI_MODEL",
                getattr(config, "codex_cli_model", os.environ.get("MODEL", "gpt-5.3-codex")),
            ),
            codex_bin=os.environ.get("CODEX_CLI_BIN", getattr(config, "codex_cli_bin", "codex")),
            cwd=root,
        )
    return OpenAICompatibleProvider(
        model=os.environ.get("MODEL", getattr(config, "model", "gpt-5.5")),
        base_url=os.environ.get("OPENAI_BASE_URL", getattr(config, "openai_base_url", "https://api.openai.com/v1")),
        api_key=os.environ.get("OPENAI_API_KEY"),
        reasoning_effort=os.environ.get("REASONING_EFFORT", getattr(config, "reasoning_effort", "medium")),
    )


def _messages_to_prompt(messages: list[dict[str, str]], *, response_format: str | None = None) -> str:
    parts = []
    if response_format == "json":
        parts.append("Return only valid JSON. Do not include Markdown fences or commentary.")
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts).strip() + "\n"


def _provider_tmp_dir(cwd: Path | None) -> Path:
    base = (cwd or Path.cwd()) / ".viktor_tmp"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"codex_cli_{uuid.uuid4().hex}"
    path.mkdir()
    return path
