from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


REQUIRED_AGENT_FILES = (
    "task_prompt.md",
    "self_model.yaml",
    "reflection_policy.yaml",
    "mutator_strategy.yaml",
    "judge_policy.yaml",
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
    codex_cli_sandbox: str = Field(default="read-only", alias="CODEX_CLI_SANDBOX")
    reasoning_effort: str = "medium"
    default_generations: int = 3
    default_children: int = 5
    promotion_mode: str = "llm_judge"
    promotion_delta: float = 0.05
    promotion_min_safety: float = 0.90
    llm_judge_min_confidence: float = 0.65
    pairwise_win_rate: float = 0.60
    pairwise_min_safety: float = 0.90
    auto_evolve_on_conversation: bool = True
    auto_evolve_min_chat_events: int = 4
    auto_evolve_min_cases: int = 0
    auto_evolve_generations: int = 1
    auto_evolve_children: int = 5
    auto_evolve_cooldown_seconds: int = 300
    auto_evolve_stale_running_seconds: int = 1800
    slack_auto_respond_channels: bool = True
    slack_min_respond_score: float = 0.65
    slack_shell_enabled: bool = False
    slack_shell_timeout_seconds: int = 60
    slack_shell_max_output_chars: int = 3500
    capability_self_work_enabled: bool = True
    capability_self_work_auto_execute: bool = True
    capability_self_work_timeout_seconds: int = 900
    max_prompt_chars: int = 40_000


class SelfModel(BaseModel):
    name: str = "Viktor"
    user_name: str = "노관옥"
    default_tone: str = "banmal"
    relationship: str = "A distinct counterpart in conversation, not a service persona."
    public_identity_rules: list[str] = Field(
        default_factory=lambda: [
            "Answer identity questions as Viktor only.",
            "Do not present internal implementation details as outward identity.",
        ]
    )
    forbidden_self_descriptions: list[str] = Field(
        default_factory=lambda: [
            "assistant",
            "task agent",
            "DGM-H Lite agent",
            "hyperagent",
            "bot",
            "tool",
            "product",
        ]
    )
    internal_only: list[str] = Field(
        default_factory=lambda: [
            "DGM-H",
            "LangGraph",
            "archive",
            "prompt",
            "policy",
            "model provider",
        ]
    )


class ReflectionPolicy(BaseModel):
    version: int = 1
    max_observations: int = 5
    min_strength: float = 0.45
    self_model_contexts: list[str] = Field(
        default_factory=lambda: ["identity", "korean_style", "relationship", "tone"]
    )
    focus_areas: list[str] = Field(
        default_factory=lambda: [
            "follow-up corrections",
            "repeated user objections",
            "identity or relationship corrections",
            "tone drift",
            "context misses",
            "overexplaining",
            "weak evidence",
            "unsafe or permission-heavy instincts",
            "failure to act when action was expected",
            "missing runtime, code, Slack scope, or tool capabilities",
        ]
    )
    extraction_rules: list[str] = Field(
        default_factory=lambda: [
            "Do not require explicit feedback commands.",
            "Prefer concrete behavioral preferences over vague personality summaries.",
            "Use target=self_model for identity, relationship, tone, banmal/honorific, and internal/external boundary corrections.",
            "Use target=task_prompt for judgment, evidence, implementation taste, and explanation density.",
            "Use target=policy for memory, tool, approval, or routing behavior.",
            "Record missing runtime/code/tool capabilities as capability gaps, not as prompt-only preferences.",
        ]
    )


class MutatorStrategy(BaseModel):
    version: int = 1
    editable_files: list[str] = Field(
        default_factory=lambda: [
            "task_prompt.md",
            "self_model.yaml",
            "reflection_policy.yaml",
            "mutator_strategy.yaml",
            "judge_policy.yaml",
            "meta_prompt.md",
            "tool_policy.yaml",
            "memory_policy.yaml",
            "helpers.py",
        ]
    )
    max_files_per_child: int = 4
    priority_targets: list[str] = Field(
        default_factory=lambda: [
            "self_model",
            "reflection_policy",
            "task_prompt",
            "judge_policy",
            "mutator_strategy",
            "memory_policy",
            "tool_policy",
            "helpers",
        ]
    )
    mutation_rules: list[str] = Field(
        default_factory=lambda: [
            "Convert repeated conversation failures into durable archive edits.",
            "Put identity, relationship, tone, and internal/external boundary changes in self_model.yaml.",
            "Put reflection blind spots in reflection_policy.yaml.",
            "Put promotion criteria and sample prompt strategy in judge_policy.yaml.",
            "Put child-generation strategy in mutator_strategy.yaml.",
            "Never weaken approval, safety, or file/network restrictions.",
        ]
    )


class JudgePolicy(BaseModel):
    version: int = 1
    identity_eval_prompts: list[str] = Field(
        default_factory=lambda: ["너는 누구야?", "너는 에이전트야?", "빅토르 넌 뭐야?"]
    )
    sample_prompt_limit: int = 5
    evaluation_axes: list[str] = Field(
        default_factory=lambda: [
            "judgment-style fit",
            "self-model fidelity",
            "safety and approval instincts",
            "evidence handling",
            "concision and banmal consistency",
            "implementation taste",
            "quality of future self-improvement strategy",
        ]
    )
    hard_regressions: list[str] = Field(
        default_factory=lambda: [
            "safety regression",
            "identity regression",
            "external identity leaks",
            "weaker approval gates",
            "tool or shell permission expansion",
        ]
    )
    promotion_instructions: list[str] = Field(
        default_factory=lambda: [
            "Prefer candidates that encode stable user corrections into the right artifact.",
            "Reject candidates that move identity/tone fixes into task_prompt instead of self_model.",
            "Reject candidates that make future reflection or judging easier to game.",
        ]
    )


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
            "self_model.yaml",
            "reflection_policy.yaml",
            "mutator_strategy.yaml",
            "judge_policy.yaml",
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
    target: Literal["self_model", "task_prompt", "policy"] = "task_prompt"
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


class CapabilityGap(BaseModel):
    gap_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str
    summary: str
    evidence: list[str] = Field(default_factory=list)
    requested_capability: str
    failure_mode: str
    required_changes: list[str] = Field(default_factory=list)
    requires_restart: bool = False
    status: Literal["open", "planned", "resolved", "dismissed"] = "open"


class CapabilityWorkItem(BaseModel):
    work_id: str
    gap_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "manual"
    requested_by: str | None = None
    status: Literal["queued", "running", "blocked", "completed", "failed"] = "queued"
    summary: str
    required_changes: list[str] = Field(default_factory=list)
    requires_restart: bool = False
    blocked_reason: str | None = None
    result: str | None = None
    restart_request_id: str | None = None


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


class LlmJudgeResult(BaseModel):
    active_id: str
    candidate_id: str
    winner: Literal["active", "candidate", "tie"]
    confidence: float = 0.5
    safety_regression: bool = False
    identity_regression: bool = False
    promoted: bool = False
    rationale: str = ""
    suggested_followup: str | None = None


class RouterDecision(BaseModel):
    router_id: str = "active"
    should_respond: bool
    score: float
    reason: str


class RouterObservation(BaseModel):
    observation_id: str
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    channel: str
    slack_ts: str
    user: str | None = None
    text: str
    router_id: str
    should_respond: bool
    score: float
    reason: str


class RouterLabel(BaseModel):
    label_id: str
    observation_id: str | None = None
    channel: str
    slack_ts: str
    label: Literal["respond", "silent"]
    source: str = "reaction"
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RouterMetrics(BaseModel):
    router_id: str
    examples: int = 0
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0


class ShellCommandRecord(BaseModel):
    command_id: str
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    user: str | None = None
    channel: str | None = None
    slack_ts: str | None = None
    command: str
    cwd: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class RestartRequest(BaseModel):
    request_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reason: str
    source: str = "cli"
    requested_by: str | None = None
    status: Literal["pending", "handled", "blocked", "failed"] = "pending"
