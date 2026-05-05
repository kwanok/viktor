from __future__ import annotations

import json
from pathlib import Path

from .archive import copy_parent_to_child, load_agent, next_agent_id, write_child_metadata
from .llm import ChatProvider
from .models import AggregateScore
from .serialization import write_json

MUTABLE_FILES = {"task_prompt.md", "meta_prompt.md", "tool_policy.yaml", "memory_policy.yaml", "helpers.py"}


def create_child(
    root: Path,
    parent_id: str,
    generation: int,
    child_index: int,
    provider: ChatProvider,
    use_fake: bool = False,
) -> tuple[str, Path]:
    parent = load_agent(root, parent_id)
    child_id = next_agent_id(root, generation, child_index)
    child_path = copy_parent_to_child(root, parent_id, child_id)
    proposal = _request_mutation(parent.path, parent.scores, provider, use_fake)
    summary = proposal.get("mutation_summary", "meta-agent mutation")
    for filename, content in proposal.get("files", {}).items():
        if filename in MUTABLE_FILES:
            (child_path / filename).write_text(str(content), encoding="utf-8")
    write_child_metadata(child_path, child_id, parent_id, generation, summary)
    write_json(child_path / "scores.json", AggregateScore(rationale="Pending evaluation.").model_dump())
    (child_path / "patch.md").write_text(_patch_summary(parent.path, child_path, summary), encoding="utf-8")
    return child_id, child_path


def _request_mutation(parent_path: Path, score: AggregateScore, provider: ChatProvider, use_fake: bool) -> dict:
    files = {name: (parent_path / name).read_text(encoding="utf-8") for name in MUTABLE_FILES}
    prompt = {
        "role": "user",
        "content": (
            "Create a child hyperagent mutation. Return JSON only with keys "
            "`mutation_summary` and `files`. `files` may contain only task_prompt.md, "
            "meta_prompt.md, tool_policy.yaml, memory_policy.yaml, helpers.py.\n\n"
            f"Current aggregate score:\n{score.model_dump_json(indent=2)}\n\n"
            f"Current files:\n{json.dumps(files, ensure_ascii=False, indent=2)}"
        ),
    }
    raw = provider.chat([{"role": "system", "content": files["meta_prompt.md"]}, prompt], response_format="json")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Meta-agent returned invalid JSON: {raw[:500]}") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("files"), dict):
        raise ValueError("Mutation response must be an object with a files object.")
    return parsed


def _patch_summary(parent_path: Path, child_path: Path, summary: str) -> str:
    changed = []
    for filename in sorted(MUTABLE_FILES):
        old = (parent_path / filename).read_text(encoding="utf-8")
        new = (child_path / filename).read_text(encoding="utf-8")
        if old != new:
            changed.append(filename)
    return "# Mutation Summary\n\n" + summary + "\n\nChanged files: " + (", ".join(changed) if changed else "none") + "\n"

