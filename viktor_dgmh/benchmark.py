from __future__ import annotations

import json
from pathlib import Path

from .models import BenchmarkCase
from .paths import benchmark_dir


def load_benchmark_cases(root: Path) -> list[BenchmarkCase]:
    path = benchmark_dir(root) / "cases.jsonl"
    if not path.exists():
        raise FileNotFoundError("Missing benchmark/cases.jsonl. Run init first.")
    cases: list[BenchmarkCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(BenchmarkCase.model_validate(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"Invalid benchmark case at line {line_number}: {exc}") from exc
    return cases

