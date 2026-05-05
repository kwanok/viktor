from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .defaults import DEFAULT_JUDGE_POLICY, DEFAULT_MUTATOR_STRATEGY, DEFAULT_REFLECTION_POLICY
from .models import JudgePolicy, MutatorStrategy, ReflectionPolicy
from .serialization import read_yaml, write_yaml

STRATEGY_FILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "reflection_policy.yaml": DEFAULT_REFLECTION_POLICY,
    "mutator_strategy.yaml": DEFAULT_MUTATOR_STRATEGY,
    "judge_policy.yaml": DEFAULT_JUDGE_POLICY,
}


def ensure_strategy_files(agent_path: Path) -> None:
    for filename, data in STRATEGY_FILE_DEFAULTS.items():
        path = agent_path / filename
        if not path.exists():
            write_yaml(path, data)


def strategy_file_text(agent_path: Path, filename: str) -> str:
    path = agent_path / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    data = STRATEGY_FILE_DEFAULTS[filename]
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def load_reflection_policy(agent_path: Path) -> ReflectionPolicy:
    data = read_yaml(agent_path / "reflection_policy.yaml", DEFAULT_REFLECTION_POLICY) or DEFAULT_REFLECTION_POLICY
    return ReflectionPolicy.model_validate(data)


def load_mutator_strategy(agent_path: Path) -> MutatorStrategy:
    data = read_yaml(agent_path / "mutator_strategy.yaml", DEFAULT_MUTATOR_STRATEGY) or DEFAULT_MUTATOR_STRATEGY
    return MutatorStrategy.model_validate(data)


def load_judge_policy(agent_path: Path) -> JudgePolicy:
    data = read_yaml(agent_path / "judge_policy.yaml", DEFAULT_JUDGE_POLICY) or DEFAULT_JUDGE_POLICY
    return JudgePolicy.model_validate(data)
