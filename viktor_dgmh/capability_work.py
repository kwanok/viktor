from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from .llm import CodexCliProvider
from .memory import (
    append_capability_work_item,
    load_capability_gaps,
    load_capability_work_items,
    load_latest_capability_work_items,
)
from .models import CapabilityGap, CapabilityWorkItem, Config
from .runtime_lifecycle import request_restart

ACTIVE_WORK_STATUSES = {"queued", "running", "blocked"}
EXECUTABLE_WORK_STATUSES = {"queued", "failed"}
EXTERNAL_APPROVAL_CHANGES = {
    "slack_scope",
    "oauth_scope",
    "oauth",
    "account_permission",
    "permission",
    "token",
    "secret",
}


def request_capability_work(
    root: Path,
    gap: CapabilityGap,
    *,
    source: str = "manual",
    requested_by: str | None = None,
) -> CapabilityWorkItem:
    existing = latest_work_for_gap(root, gap.gap_id)
    if existing and existing.status in ACTIVE_WORK_STATUSES:
        return existing

    blocked_reason = external_approval_blocker(gap)
    item = CapabilityWorkItem(
        work_id=f"work_{uuid.uuid4().hex}",
        gap_id=gap.gap_id,
        updated_at=_now(),
        source=source,
        requested_by=requested_by,
        status="blocked" if blocked_reason else "queued",
        summary=f"Turn capability gap into validated Viktor Agent work: {gap.summary}",
        required_changes=gap.required_changes,
        requires_restart=gap.requires_restart,
        blocked_reason=blocked_reason,
    )
    append_capability_work_item(root, item)
    return item


def approve_capability_work(
    root: Path,
    item: CapabilityWorkItem,
    *,
    approved_by: str | None = None,
    evidence: str = "",
) -> CapabilityWorkItem:
    if item.status != "blocked":
        return item
    updated = item.model_copy(
        update={
            "updated_at": _now(),
            "requested_by": approved_by or item.requested_by,
            "status": "queued",
            "blocked_reason": None,
            "result": f"External approval recorded. {evidence}".strip(),
        }
    )
    append_capability_work_item(root, updated)
    return updated


def find_work_ready_for_approval(root: Path, text: str, thread_context: str | None = None) -> CapabilityWorkItem | None:
    if not looks_like_external_approval(text, thread_context):
        return None
    gaps_by_id = {gap.gap_id: gap for gap in load_capability_gaps(root)}
    candidates = []
    for item in load_latest_capability_work_items(root):
        if item.status != "blocked":
            continue
        gap = gaps_by_id.get(item.gap_id)
        if gap and _message_targets_gap(text, thread_context, gap):
            candidates.append(item)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.updated_at)[-1]


def approve_work_from_message(
    root: Path,
    text: str,
    thread_context: str | None = None,
    *,
    approved_by: str | None = None,
) -> CapabilityWorkItem | None:
    item = find_work_ready_for_approval(root, text, thread_context)
    if item is None:
        return None
    return approve_capability_work(root, item, approved_by=approved_by, evidence=_approval_evidence(text))


def execute_capability_work(
    root: Path,
    work_id: str,
    config: Config,
    *,
    fake: bool = False,
) -> CapabilityWorkItem:
    item = find_latest_work(root, work_id)
    if item is None:
        raise ValueError(f"Unknown capability work item: {work_id}")
    if item.status not in EXECUTABLE_WORK_STATUSES:
        return item

    running = item.model_copy(update={"updated_at": _now(), "status": "running", "result": "Executor started."})
    append_capability_work_item(root, running)
    try:
        if fake:
            result = "Fake executor completed capability work."
        else:
            result = _run_codex_capability_executor(root, running, config)
        restart_request_id = None
        if running.requires_restart:
            restart = request_restart(
                root,
                f"capability work completed: {running.work_id}",
                source="capability_executor",
                requested_by=running.requested_by,
            )
            restart_request_id = restart.request_id
        completed = running.model_copy(
            update={
                "updated_at": _now(),
                "status": "completed",
                "result": result,
                "restart_request_id": restart_request_id,
            }
        )
        append_capability_work_item(root, completed)
        return completed
    except Exception as exc:
        failed = running.model_copy(
            update={
                "updated_at": _now(),
                "status": "failed",
                "result": f"{type(exc).__name__}: {exc}",
            }
        )
        append_capability_work_item(root, failed)
        return failed


def find_latest_work(root: Path, work_id: str) -> CapabilityWorkItem | None:
    items = [item for item in load_capability_work_items(root) if item.work_id == work_id]
    if not items:
        return None
    return sorted(items, key=lambda item: item.updated_at)[-1]


def latest_work_for_gap(root: Path, gap_id: str) -> CapabilityWorkItem | None:
    items = [item for item in load_latest_capability_work_items(root) if item.gap_id == gap_id]
    if not items:
        return None
    return sorted(items, key=lambda item: item.updated_at)[-1]


def external_approval_blocker(gap: CapabilityGap) -> str | None:
    normalized_changes = {_normalize_change(change) for change in gap.required_changes}
    if normalized_changes.intersection(EXTERNAL_APPROVAL_CHANGES):
        return "requires explicit external approval before Viktor can apply or restart this capability"

    text = " ".join([gap.requested_capability, gap.failure_mode, gap.summary, *gap.required_changes]).lower()
    if "slack" in text and any(token in text for token in ["scope", "oauth", "permission", "token"]):
        return "requires explicit Slack permission approval before Viktor can apply or restart this capability"
    return None


def looks_like_external_approval(text: str, thread_context: str | None = None) -> bool:
    combined = "\n".join(part for part in [thread_context or "", text] if part)
    lowered = combined.lower()
    has_approval = (
        "approved" in lowered
        or "permission added" in lowered
        or "scope added" in lowered
        or "reactions:write" in lowered
        or any(token in combined for token in ["권한 추가", "권한 열", "승인", "추가했고", "추가했어"])
    )
    asks_work = (
        "execute" in lowered
        or "implement" in lowered
        or "do the work" in lowered
        or any(token in combined for token in ["직접 작업", "작업해", "작업해서", "진행", "적용", "결과 알려"])
    )
    return has_approval and asks_work


def _message_targets_gap(text: str, thread_context: str | None, gap: CapabilityGap) -> bool:
    combined = "\n".join(part for part in [thread_context or "", text] if part).lower()
    gap_text = " ".join([gap.requested_capability, gap.failure_mode, gap.summary]).lower()
    if "reactions:write" in combined and ("reaction" in gap_text or "slack_reaction" in gap_text):
        return True
    if ":eyes:" in combined and (":eyes:" in gap_text or "reaction" in gap_text):
        return True
    return any(token in combined for token in ["slack", "reaction"]) and "slack" in gap_text


def _approval_evidence(text: str) -> str:
    cleaned = " ".join(text.split())
    return f"message={cleaned[:500]}"


def _run_codex_capability_executor(root: Path, item: CapabilityWorkItem, config: Config) -> str:
    provider = CodexCliProvider(
        model=config.codex_cli_model,
        codex_bin=config.codex_cli_bin,
        sandbox="workspace-write",
        cwd=root,
        timeout_seconds=config.capability_self_work_timeout_seconds,
    )
    prompt = _executor_prompt(item)
    return provider.chat([{"role": "user", "content": prompt}], model=config.codex_cli_model)


def _executor_prompt(item: CapabilityWorkItem) -> str:
    return f"""You are Viktor Agent's capability executor.

Execute this capability work item inside the current repository.

Work id: {item.work_id}
Gap id: {item.gap_id}
Summary: {item.summary}
Required changes: {", ".join(item.required_changes) if item.required_changes else "-"}
Requires restart after success: {item.requires_restart}

Rules:
- Implement the minimal code and tests needed for this capability.
- Do not change external accounts, secrets, or OAuth settings unless the work item already records approval.
- Do not edit archive/ACTIVE or runtime state except through project CLI commands.
- Run the relevant tests before finishing.
- If code changes require a live reload, rely on the project runtime restart request mechanism after the executor completes.
- Report concise results and any remaining blocker.
"""


def _normalize_change(change: str) -> str:
    return change.strip().lower().replace("-", "_").replace(" ", "_")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
