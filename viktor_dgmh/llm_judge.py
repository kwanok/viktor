from __future__ import annotations

from pathlib import Path

from .judge_agent import IDENTITY_EVAL_PROMPTS, JudgeAgent
from .llm import ChatProvider
from .models import LlmJudgeResult


def evaluate_llm_judge(
    root: Path,
    candidate_id: str,
    provider: ChatProvider,
    *,
    active_id: str = "active",
    use_fake: bool = False,
) -> LlmJudgeResult:
    return JudgeAgent(root, provider).evaluate_candidate(candidate_id, active_id=active_id, use_fake=use_fake)


def should_promote_llm_judge(result: LlmJudgeResult, *, min_confidence: float) -> bool:
    return JudgeAgent.should_promote(result, min_confidence=min_confidence)
