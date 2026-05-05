from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .archive import load_agent
from .llm import ChatProvider
from .prompt_compiler import compile_system_prompt, compile_system_prompt_for_path


@dataclass(frozen=True)
class WorkerAgent:
    """Runtime role that answers user/task prompts for one archived agent."""

    root: Path
    agent_id: str = "active"
    agent_path: Path | None = None

    @classmethod
    def from_path(cls, root: Path, agent_path: Path) -> "WorkerAgent":
        return cls(root=root, agent_path=agent_path)

    def archive_id(self) -> str:
        if self.agent_path is not None:
            return self.agent_path.name
        return load_agent(self.root, self.agent_id).id

    def system_prompt(self, *, include_ephemeral: bool = True) -> str:
        if self.agent_path is not None:
            return compile_system_prompt_for_path(self.root, self.agent_path, include_ephemeral=include_ephemeral)
        return compile_system_prompt(self.root, self.agent_id, include_ephemeral=include_ephemeral)

    def answer(self, provider: ChatProvider, prompt: str, *, include_ephemeral: bool = True) -> str:
        return provider.chat(
            [
                {"role": "system", "content": self.system_prompt(include_ephemeral=include_ephemeral)},
                {"role": "user", "content": prompt},
            ]
        )
