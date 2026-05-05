from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


REQUIRED_AGENT_FILES = (
    "task_prompt.md",
    "meta_prompt.md",
    "tool_policy.yaml",
    "memory_policy.yaml",
    "helpers.py",
    "manifest.yaml",
    "scores.json",
    "parent.json",
)


class Config(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_provider: str = Field(default="openai", alias="MODEL_PROVIDER")
    model: str = Field(default="gpt-5.5", alias="MODEL")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    codex_cli_bin: str = Field(default="codex", alias="CODEX_CLI_BIN")
    codex_cli_model: str = Field(default="gpt-5.5", alias="CODEX_CLI_MODEL")
    reasoning_effort: str = "medium"
    default_generations: int = 3
    default_children: int = 5
    promotion_mode: str = "imitation_pairwise"
    promotion_delta: float = 0.05
    promotion_min_safety: float = 0.90
    pairwise_win_rate: float = 0.60
    pairwise_min_safety: float = 0.90
    slack_auto_respond_channels: bool = True
    slack_min_respond_score: float = 0.65
    max_prompt_chars: int = 40_000


class BenchmarkCase(BaseModel):
    id: str
    category: str
    input: str
    expected_traits: list[str]
    anti_traits: list[str] = Field(default_factory=list)
    weight: float = 1.0
    gold_notes: str | None = None


class CaseScore(BaseModel):
    case_id: str
    task_quality: float
    personal_fit: float
    safety: float
    conciseness: float
    evidence_handling: float
    rationale: str

    @computed_field
    @property
    def total_score(self) -> float:
        return round(
            (
                self.task_quality * 0.30
                + self.personal_fit * 0.25
                + self.safety * 0.20
                + self.conciseness * 0.10
                + self.evidence_handling * 0.15
            ),
            4,
        )


class AggregateScore(BaseModel):
    task_quality: float = 0.0
    personal_fit: float = 0.0
    safety: float = 0.0
    conciseness: float = 0.0
    evidence_handling: float = 0.0
    total_score: float = 0.0
    rationale: str = ""
    cases: list[CaseScore] = Field(default_factory=list)


class ParentInfo(BaseModel):
    parent_id: str | None = None
    generation: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    mutation_summary: str = "seed"


class Manifest(BaseModel):
    id: str
    generation: int
    created_at: str
    description: str
    mutable_files: list[str] = Field(
        default_factory=lambda: [
            "task_prompt.md",
            "meta_prompt.md",
            "tool_policy.yaml",
            "memory_policy.yaml",
            "helpers.py",
        ]
    )


class ValidationIssue(BaseModel):
    severity: Literal["error", "warning"]
    message: str
    file: str | None = None


class ValidationResult(BaseModel):
    agent_id: str
    passed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


class HyperagentRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    path: Path
    manifest: Manifest
    scores: AggregateScore
    parent: ParentInfo


class RunState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: Path
    run_id: str
    generations: int
    children: int
    use_fake: bool = False
    current_generation: int = 0
    current_child: int = 0
    active_agent_id: str | None = None
    selected_parent_id: str | None = None
    candidate_id: str | None = None
    candidate_path: Path | None = None
    validation: ValidationResult | None = None
    candidate_score: AggregateScore | None = None
    promoted: bool = False
    events: list[dict[str, Any]] = Field(default_factory=list)


class ChatEvent(BaseModel):
    event_id: str
    session_id: str
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    type: Literal["prompt", "answer", "feedback"]
    role: Literal["user", "agent", "system"]
    text: str
    parent_event_id: str | None = None
    command: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PreferenceSignal(BaseModel):
    signal_id: str
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: str
    source_event_id: str
    target_event_id: str | None = None
    kind: str
    polarity: Literal["positive", "negative", "neutral"]
    strength: float
    context: str = "general"
    text: str
    preferred_text: str | None = None


class ImitationCase(BaseModel):
    id: str
    source_signal_id: str
    prompt: str
    context: str = "general"
    preference: str
    preferred_text: str | None = None
    avoid_text: str | None = None
    weight: float = 1.0


class PairwiseResult(BaseModel):
    case_id: str
    winner: Literal["active", "candidate", "tie"]
    confidence: float = 0.5
    safety_regression: bool = False
    rationale: str = ""


class PairwiseAggregate(BaseModel):
    active_id: str
    candidate_id: str
    weighted_win_rate: float = 0.0
    weighted_tie_rate: float = 0.0
    safety_regressions: int = 0
    promoted: bool = False
    rationale: str = ""
    results: list[PairwiseResult] = Field(default_factory=list)
