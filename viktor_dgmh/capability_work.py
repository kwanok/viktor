from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from .memory import append_capability_work_item, load_capability_work_items
from .models import CapabilityGap, CapabilityWorkItem

ACTIVE_WORK_STATUSES = {"queued", "running", "blocked"}
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


def latest_work_for_gap(root: Path, gap_id: str) -> CapabilityWorkItem | None:
    items = [item for item in load_capability_work_items(root) if item.gap_id == gap_id]
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


def _normalize_change(change: str) -> str:
    return change.strip().lower().replace("-", "_").replace(" ", "_")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
