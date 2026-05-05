from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from .defaults import (
    DEFAULT_BENCHMARK_CASES,
    DEFAULT_CONFIG,
    SEED_HELPERS,
    SEED_MEMORY_POLICY,
    SEED_META_PROMPT,
    SEED_TASK_PROMPT,
    SEED_TOOL_POLICY,
)
from .models import AggregateScore, HyperagentRecord, Manifest, ParentInfo
from .paths import active_path, archive_dir, benchmark_dir, config_path, memory_dir, router_archive_dir, router_active_path, router_dir, runs_dir
from .serialization import read_json, read_yaml, write_json, write_yaml


def init_workspace(root: Path, force: bool = False) -> str:
    root.mkdir(parents=True, exist_ok=True)
    archive_dir(root).mkdir(parents=True, exist_ok=True)
    benchmark_dir(root).mkdir(parents=True, exist_ok=True)
    runs_dir(root).mkdir(parents=True, exist_ok=True)
    memory_dir(root).mkdir(parents=True, exist_ok=True)
    router_dir(root).mkdir(parents=True, exist_ok=True)
    router_archive_dir(root).mkdir(parents=True, exist_ok=True)

    if not config_path(root).exists() or force:
        write_yaml(config_path(root), DEFAULT_CONFIG)

    if not router_active_path(root).exists() or force:
        from .router import DEFAULT_ROUTER_POLICY

        write_yaml(router_active_path(root), DEFAULT_ROUTER_POLICY)

    cases_path = benchmark_dir(root) / "cases.jsonl"
    if not cases_path.exists() or force:
        cases_path.write_text(
            "".join(__import__("json").dumps(case, ensure_ascii=False) + "\n" for case in DEFAULT_BENCHMARK_CASES),
            encoding="utf-8",
        )

    seed_id = "gen000_seed"
    seed_path = archive_dir(root) / seed_id
    if seed_path.exists() and not force:
        if not active_path(root).exists():
            active_path(root).write_text(seed_id + "\n", encoding="utf-8")
        return seed_id

    if seed_path.exists():
        shutil.rmtree(seed_path)
    seed_path.mkdir(parents=True)

    created_at = datetime.now(timezone.utc).isoformat()
    (seed_path / "task_prompt.md").write_text(SEED_TASK_PROMPT, encoding="utf-8")
    (seed_path / "meta_prompt.md").write_text(SEED_META_PROMPT, encoding="utf-8")
    write_yaml(seed_path / "tool_policy.yaml", SEED_TOOL_POLICY)
    write_yaml(seed_path / "memory_policy.yaml", SEED_MEMORY_POLICY)
    (seed_path / "helpers.py").write_text(SEED_HELPERS, encoding="utf-8")
    write_yaml(
        seed_path / "manifest.yaml",
        Manifest(
            id=seed_id,
            generation=0,
            created_at=created_at,
            description="Seed personal DGM-H Lite hyperagent.",
        ).model_dump(),
    )
    write_json(seed_path / "scores.json", AggregateScore(total_score=0.5, safety=1.0, rationale="Seed baseline.").model_dump())
    write_json(seed_path / "parent.json", ParentInfo(parent_id=None, generation=0, mutation_summary="seed").model_dump())
    active_path(root).write_text(seed_id + "\n", encoding="utf-8")
    return seed_id


def load_config(root: Path) -> dict:
    return read_yaml(config_path(root), DEFAULT_CONFIG)


def get_active_agent_id(root: Path) -> str:
    path = active_path(root)
    if not path.exists():
        raise FileNotFoundError("No ACTIVE hyperagent. Run `python -m viktor_dgmh init` first.")
    return path.read_text(encoding="utf-8").strip()


def set_active_agent(root: Path, agent_id: str) -> None:
    if not (archive_dir(root) / agent_id).is_dir():
        raise FileNotFoundError(f"Unknown hyperagent: {agent_id}")
    active_path(root).write_text(agent_id + "\n", encoding="utf-8")


def load_agent(root: Path, agent_id: str) -> HyperagentRecord:
    if agent_id == "active":
        agent_id = get_active_agent_id(root)
    path = archive_dir(root) / agent_id
    if not path.is_dir():
        raise FileNotFoundError(f"Unknown hyperagent: {agent_id}")
    manifest = Manifest.model_validate(read_yaml(path / "manifest.yaml"))
    scores = AggregateScore.model_validate(read_json(path / "scores.json", {}))
    parent = ParentInfo.model_validate(read_json(path / "parent.json", {}))
    return HyperagentRecord(id=agent_id, path=path, manifest=manifest, scores=scores, parent=parent)


def list_agents(root: Path) -> list[HyperagentRecord]:
    records: list[HyperagentRecord] = []
    base = archive_dir(root)
    if not base.exists():
        return records
    for child in sorted(path for path in base.iterdir() if path.is_dir()):
        try:
            records.append(load_agent(root, child.name))
        except Exception:
            continue
    return records


def next_agent_id(root: Path, generation: int, child_index: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return f"gen{generation:03d}_child{child_index:03d}_{stamp}"


def copy_parent_to_child(root: Path, parent_id: str, child_id: str) -> Path:
    parent_path = archive_dir(root) / parent_id
    child_path = archive_dir(root) / child_id
    if child_path.exists():
        raise FileExistsError(f"Candidate already exists: {child_id}")
    shutil.copytree(parent_path, child_path)
    return child_path


def write_child_metadata(child_path: Path, child_id: str, parent_id: str, generation: int, summary: str) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    write_yaml(
        child_path / "manifest.yaml",
        Manifest(
            id=child_id,
            generation=generation,
            created_at=created_at,
            description=summary,
        ).model_dump(),
    )
    write_json(
        child_path / "parent.json",
        ParentInfo(parent_id=parent_id, generation=generation, mutation_summary=summary).model_dump(),
    )


def persist_score(child_path: Path, score: AggregateScore) -> None:
    write_json(child_path / "scores.json", score.model_dump())
