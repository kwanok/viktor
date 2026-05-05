from __future__ import annotations

from pathlib import Path

from .llm import ChatProvider
from .meta_agent import MUTABLE_FILES, MetaAgent
from .models import AggregateScore


def create_child(
    root: Path,
    parent_id: str,
    generation: int,
    child_index: int,
    provider: ChatProvider,
    use_fake: bool = False,
) -> tuple[str, Path]:
    return MetaAgent(root).create_child(parent_id, generation, child_index, provider, use_fake)


def _request_mutation(root: Path, parent_path: Path, score: AggregateScore, provider: ChatProvider, use_fake: bool) -> dict:
    return MetaAgent(root).request_mutation(parent_path, score, provider, use_fake)


def _build_evolution_brief(root: Path) -> dict:
    return MetaAgent(root).build_evolution_brief()


def _patch_summary(parent_path: Path, child_path: Path, summary: str) -> str:
    return MetaAgent(parent_path.parents[1]).patch_summary(parent_path, child_path, summary)
