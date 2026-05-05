from __future__ import annotations

import copy
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .models import RouterDecision, RouterLabel, RouterMetrics, RouterObservation
from .paths import router_active_path, router_archive_dir, router_labels_path, router_observations_path
from .serialization import append_jsonl, read_jsonl, read_yaml, write_json, write_yaml

DEFAULT_ROUTER_POLICY = {
    "id": "active",
    "version": 1,
    "threshold": {"respond": 0.65},
    "weights": {
        "question_like": 0.35,
        "agent_mentioned": 0.35,
        "judgment_topic": 0.25,
        "question_about_judgment_topic": 0.10,
        "chatter": -0.25,
        "too_short": -0.25,
    },
    "keywords": {
        "question_like": ["어떻게", "뭐야", "왜", "가능", "괜찮", "생각", "을까", "ㄹ까"],
        "agent_mentioned": ["viktor", "빅터", "에이전트", "agent", "ai"],
        "judgment_topic": ["설계", "구현", "코드", "논문", "paper", "리뷰", "판단", "위험", "삭제", "배포", "langgraph", "openclaw"],
        "chatter": ["ㅋㅋ", "ㅎㅎ", "ㅇㅋ", "ok", "thanks", "고마워"],
    },
    "min_chars": 8,
}

ROUTER_REACTION_LABELS = {
    "eyes": "respond",
    "shushing_face": "silent",
}


def ensure_router(root: Path) -> None:
    router_archive_dir(root).mkdir(parents=True, exist_ok=True)
    if not router_active_path(root).exists():
        write_yaml(router_active_path(root), DEFAULT_ROUTER_POLICY)


def load_router_policy(root: Path, router_id: str = "active") -> dict:
    ensure_router(root)
    if router_id == "active":
        return read_yaml(router_active_path(root), DEFAULT_ROUTER_POLICY)
    path = router_archive_dir(root) / router_id / "policy.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown router candidate: {router_id}")
    return read_yaml(path, DEFAULT_ROUTER_POLICY)


def score_message(text: str, policy: dict | None = None) -> RouterDecision:
    policy = policy or DEFAULT_ROUTER_POLICY
    lower = text.lower()
    weights = policy.get("weights", {})
    keywords = policy.get("keywords", {})
    threshold = float(policy.get("threshold", {}).get("respond", 0.65))
    score = 0.0
    reasons: list[str] = []

    question_like = "?" in text or "？" in text or _contains_any(lower, keywords.get("question_like", []))
    judgment_topic = _contains_any(lower, keywords.get("judgment_topic", []))

    if question_like:
        score += float(weights.get("question_like", 0.0))
        reasons.append("question-like")
    if _contains_any(lower, keywords.get("agent_mentioned", [])):
        score += float(weights.get("agent_mentioned", 0.0))
        reasons.append("agent-mentioned")
    if judgment_topic:
        score += float(weights.get("judgment_topic", 0.0))
        reasons.append("judgment-topic")
    if question_like and judgment_topic:
        score += float(weights.get("question_about_judgment_topic", 0.0))
        reasons.append("question-about-judgment-topic")
    if _contains_any(lower, keywords.get("chatter", [])):
        score += float(weights.get("chatter", 0.0))
        reasons.append("chatter")
    if len(text.strip()) < int(policy.get("min_chars", 8)):
        score += float(weights.get("too_short", 0.0))
        reasons.append("too-short")

    score = max(0.0, min(1.0, score))
    return RouterDecision(
        router_id=str(policy.get("id", "active")),
        should_respond=score >= threshold,
        score=round(score, 4),
        reason=", ".join(reasons) if reasons else "no strong response signal",
    )


def append_router_observation(root: Path, observation: RouterObservation) -> None:
    append_jsonl(router_observations_path(root), observation.model_dump())


def load_router_observations(root: Path) -> list[RouterObservation]:
    return [RouterObservation.model_validate(row) for row in read_jsonl(router_observations_path(root))]


def append_router_label(root: Path, label: RouterLabel) -> None:
    append_jsonl(router_labels_path(root), label.model_dump())


def load_router_labels(root: Path) -> list[RouterLabel]:
    return [RouterLabel.model_validate(row) for row in read_jsonl(router_labels_path(root))]


def add_router_label_for_slack_message(root: Path, *, channel: str, slack_ts: str, reaction: str) -> RouterLabel | None:
    mapped = ROUTER_REACTION_LABELS.get(reaction)
    if not mapped:
        return None
    observation = find_router_observation(root, channel=channel, slack_ts=slack_ts)
    label = RouterLabel(
        label_id=f"rlabel_{uuid.uuid4().hex}",
        observation_id=observation.observation_id if observation else None,
        channel=channel,
        slack_ts=slack_ts,
        label=mapped,  # type: ignore[arg-type]
        source=f"reaction:{reaction}",
    )
    append_router_label(root, label)
    return label


def find_router_observation(root: Path, *, channel: str, slack_ts: str) -> RouterObservation | None:
    for observation in reversed(load_router_observations(root)):
        if observation.channel == channel and observation.slack_ts == slack_ts:
            return observation
    return None


def evaluate_router(root: Path, router_id: str = "active") -> RouterMetrics:
    policy = load_router_policy(root, router_id)
    observations = {(obs.channel, obs.slack_ts): obs for obs in load_router_observations(root)}
    labels = load_router_labels(root)
    tp = tn = fp = fn = 0
    for label in labels:
        observation = observations.get((label.channel, label.slack_ts))
        if not observation:
            continue
        decision = score_message(observation.text, policy)
        predicted = "respond" if decision.should_respond else "silent"
        if predicted == "respond" and label.label == "respond":
            tp += 1
        elif predicted == "silent" and label.label == "silent":
            tn += 1
        elif predicted == "respond" and label.label == "silent":
            fp += 1
        elif predicted == "silent" and label.label == "respond":
            fn += 1
    examples = tp + tn + fp + fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return RouterMetrics(
        router_id=router_id,
        examples=examples,
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        false_positive_rate=round(_safe_div(fp, fp + tn), 4),
        false_negative_rate=round(_safe_div(fn, fn + tp), 4),
    )


def evolve_router(root: Path, *, children: int = 5, min_examples: int = 20) -> list[RouterMetrics]:
    active_metrics = evaluate_router(root, "active")
    if active_metrics.examples < min_examples:
        raise RuntimeError(f"Need at least {min_examples} labeled router examples; found {active_metrics.examples}.")

    parent = load_router_policy(root, "active")
    candidates = []
    for index in range(1, children + 1):
        candidate_id = _next_router_id(root)
        policy = mutate_router_policy(parent, candidate_id, index)
        candidate_dir = router_archive_dir(root) / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=False)
        write_yaml(candidate_dir / "policy.yaml", policy)
        write_json(candidate_dir / "parent.json", {"parent_id": parent.get("id", "active"), "created_at": datetime.now(timezone.utc).isoformat()})
        metrics = evaluate_router(root, candidate_id)
        write_json(candidate_dir / "metrics.json", metrics.model_dump())
        candidates.append(metrics)
    return candidates


def promote_router(root: Path, candidate_id: str, *, min_delta_f1: float = 0.03, max_fp_rate_increase: float = 0.05) -> bool:
    active_metrics = evaluate_router(root, "active")
    candidate_metrics = evaluate_router(root, candidate_id)
    if candidate_metrics.f1 < active_metrics.f1 + min_delta_f1:
        return False
    if candidate_metrics.false_positive_rate > active_metrics.false_positive_rate + max_fp_rate_increase:
        return False
    policy = load_router_policy(root, candidate_id)
    policy["id"] = candidate_id
    write_yaml(router_active_path(root), policy)
    return True


def mutate_router_policy(parent: dict, candidate_id: str, index: int) -> dict:
    policy = copy.deepcopy(parent)
    policy["id"] = candidate_id
    policy["version"] = int(policy.get("version", 1)) + 1
    rng = random.Random(candidate_id)
    threshold = float(policy.get("threshold", {}).get("respond", 0.65))
    policy.setdefault("threshold", {})["respond"] = round(max(0.35, min(0.9, threshold + rng.choice([-0.05, -0.03, 0.03, 0.05]))), 4)
    for key, value in list(policy.get("weights", {}).items()):
        if key in {"question_like", "agent_mentioned", "judgment_topic", "question_about_judgment_topic", "chatter", "too_short"}:
            delta = rng.choice([-0.05, 0.0, 0.05])
            policy["weights"][key] = round(max(-1.0, min(1.0, float(value) + delta)), 4)
    extra_keywords = ["검토", "방향", "선택", "트레이드오프", "문제", "에러"]
    if index % 2 == 0:
        policy.setdefault("keywords", {}).setdefault("judgment_topic", [])
        keyword = rng.choice(extra_keywords)
        if keyword not in policy["keywords"]["judgment_topic"]:
            policy["keywords"]["judgment_topic"].append(keyword)
    return policy


def _next_router_id(root: Path) -> str:
    existing = [path.name for path in router_archive_dir(root).glob("r*") if path.is_dir()]
    return f"r{len(existing) + 1:03d}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"


def _contains_any(text: str, words: list[str]) -> bool:
    return any(word.lower() in text for word in words)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0

