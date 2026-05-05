from __future__ import annotations

import json
import os
from dataclasses import dataclass
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

