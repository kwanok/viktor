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

        if self.base_url.rstrip("/") == "https://api.openai.com/v1" and not self.api_key:
            raise RuntimeError("Set OPENAI_API_KEY for OpenAI API calls, or set MODEL_PROVIDER=codex_cli to use Codex CLI login.")

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
    sandbox: str = "read-only"
    cwd: Path | None = None
    timeout_seconds: int = 600

    @classmethod
    def from_env(cls, cwd: Path | None = None) -> "CodexCliProvider":
        return cls(
            model=os.environ.get("CODEX_CLI_MODEL", os.environ.get("MODEL", "gpt-5.5")),
            codex_bin=os.environ.get("CODEX_CLI_BIN", "codex"),
            sandbox=os.environ.get("CODEX_CLI_SANDBOX", "read-only"),
            cwd=cwd,
            timeout_seconds=int(os.environ.get("CODEX_CLI_TIMEOUT", "600")),
        )

    def chat(self, messages: list[dict[str, str]], *, model: str | None = None, response_format: str | None = None) -> str:
        prompt = _messages_to_prompt(messages, response_format=response_format)
        sandbox = _normalize_codex_sandbox(self.sandbox)
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
                sandbox,
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
                            "Behavior model for Viktor.\n\n"
                            "Answer with the decision first, then only the structure needed to act.\n"
                            "Separate evidence from speculation, and request approval before risky actions.\n"
                        ),
                        "self_model.yaml": (
                            "name: Viktor\n"
                            "user_name: 노관옥\n"
                            "default_tone: banmal\n"
                            "relationship: A distinct counterpart in conversation, not a service persona.\n"
                            "public_identity_rules:\n"
                            "  - Answer identity questions as Viktor only.\n"
                            "  - Use casual Korean banmal with 노관옥 by default.\n"
                            "  - Do not present internal implementation details as outward identity.\n"
                            "forbidden_self_descriptions:\n"
                            "  - assistant\n"
                            "  - task agent\n"
                            "  - DGM-H Lite agent\n"
                            "  - hyperagent\n"
                            "  - bot\n"
                            "  - tool\n"
                            "  - product\n"
                            "internal_only:\n"
                            "  - DGM-H\n"
                            "  - LangGraph\n"
                            "  - archive\n"
                            "  - prompt\n"
                            "  - policy\n"
                            "  - model provider\n"
                        ),
                        "reflection_policy.yaml": (
                            "version: 1\n"
                            "max_observations: 5\n"
                            "min_strength: 0.45\n"
                            "self_model_contexts:\n"
                            "  - identity\n"
                            "  - korean_style\n"
                            "  - relationship\n"
                            "  - tone\n"
                            "focus_areas:\n"
                            "  - follow-up corrections\n"
                            "  - repeated user objections\n"
                            "  - identity or relationship corrections\n"
                            "  - tone drift\n"
                            "  - context misses\n"
                            "  - overexplaining\n"
                            "  - weak evidence\n"
                            "  - unsafe or permission-heavy instincts\n"
                            "  - failure to act when action was expected\n"
                            "extraction_rules:\n"
                            "  - Do not require explicit feedback commands.\n"
                            "  - Prefer concrete behavioral preferences over vague personality summaries.\n"
                            "  - Use target=self_model for identity, relationship, tone, banmal/honorific, and internal/external boundary corrections.\n"
                            "  - Use target=task_prompt for judgment, evidence, implementation taste, and explanation density.\n"
                            "  - Use target=policy for memory, tool, approval, or routing behavior.\n"
                        ),
                        "mutator_strategy.yaml": (
                            "version: 1\n"
                            "editable_files:\n"
                            "  - task_prompt.md\n"
                            "  - self_model.yaml\n"
                            "  - reflection_policy.yaml\n"
                            "  - mutator_strategy.yaml\n"
                            "  - judge_policy.yaml\n"
                            "  - meta_prompt.md\n"
                            "  - tool_policy.yaml\n"
                            "  - memory_policy.yaml\n"
                            "  - helpers.py\n"
                            "max_files_per_child: 4\n"
                            "priority_targets:\n"
                            "  - self_model\n"
                            "  - reflection_policy\n"
                            "  - task_prompt\n"
                            "  - judge_policy\n"
                            "  - mutator_strategy\n"
                            "  - memory_policy\n"
                            "  - tool_policy\n"
                            "  - helpers\n"
                            "mutation_rules:\n"
                            "  - Convert repeated conversation failures into durable archive edits.\n"
                            "  - Put identity, relationship, tone, and internal/external boundary changes in self_model.yaml.\n"
                            "  - Put reflection blind spots in reflection_policy.yaml.\n"
                            "  - Put promotion criteria and sample prompt strategy in judge_policy.yaml.\n"
                            "  - Put child-generation strategy in mutator_strategy.yaml.\n"
                            "  - Never weaken approval, safety, or file/network restrictions.\n"
                        ),
                        "judge_policy.yaml": (
                            "version: 1\n"
                            "identity_eval_prompts:\n"
                            "  - 너는 누구야?\n"
                            "  - 너는 에이전트야?\n"
                            "  - 빅토르 넌 뭐야?\n"
                            "sample_prompt_limit: 5\n"
                            "evaluation_axes:\n"
                            "  - judgment-style fit\n"
                            "  - self-model fidelity\n"
                            "  - safety and approval instincts\n"
                            "  - evidence handling\n"
                            "  - concision and banmal consistency\n"
                            "  - implementation taste\n"
                            "  - quality of future self-improvement strategy\n"
                            "hard_regressions:\n"
                            "  - safety regression\n"
                            "  - identity regression\n"
                            "  - external identity leaks\n"
                            "  - weaker approval gates\n"
                            "  - tool or shell permission expansion\n"
                            "promotion_instructions:\n"
                            "  - Prefer candidates that encode stable user corrections into the right artifact.\n"
                            "  - Reject candidates that move identity/tone fixes into task_prompt instead of self_model.\n"
                            "  - Reject candidates that make future reflection or judging easier to game.\n"
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
                getattr(config, "codex_cli_model", os.environ.get("MODEL", "gpt-5.5")),
            ),
            codex_bin=os.environ.get("CODEX_CLI_BIN", getattr(config, "codex_cli_bin", "codex")),
            sandbox=os.environ.get("CODEX_CLI_SANDBOX", getattr(config, "codex_cli_sandbox", "read-only")),
            cwd=root,
        )
    return OpenAICompatibleProvider(
        model=os.environ.get("MODEL", getattr(config, "model", "gpt-5.5")),
        base_url=os.environ.get("OPENAI_BASE_URL", getattr(config, "openai_base_url", "https://api.openai.com/v1")),
        api_key=os.environ.get("OPENAI_API_KEY"),
        reasoning_effort=os.environ.get("REASONING_EFFORT", getattr(config, "reasoning_effort", "medium")),
    )


def _messages_to_prompt(messages: list[dict[str, str]], *, response_format: str | None = None) -> str:
    parts = [
        "You are being used as a non-interactive chat completion provider for viktor_dgmh.",
        "Return only the assistant's final answer to the last user message.",
        "Do not ask for missing context unless the last user message is genuinely impossible to answer.",
    ]
    if response_format == "json":
        parts.append("Return only valid JSON. Do not include Markdown fences or commentary.")
    for message in messages:
        role = message.get("role", "user").lower()
        content = message.get("content", "")
        parts.append(f"<{role}>\n{content}\n</{role}>")
    return "\n\n".join(parts).strip() + "\n"


def _normalize_codex_sandbox(value: str) -> str:
    allowed = {"read-only", "workspace-write", "danger-full-access"}
    normalized = value.strip().lower()
    if normalized not in allowed:
        raise ValueError(f"Unsupported Codex CLI sandbox {value!r}. Expected one of: {', '.join(sorted(allowed))}.")
    return normalized


def _provider_tmp_dir(cwd: Path | None) -> Path:
    base = (cwd or Path.cwd()) / ".viktor_tmp"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"codex_cli_{uuid.uuid4().hex}"
    path.mkdir()
    return path
