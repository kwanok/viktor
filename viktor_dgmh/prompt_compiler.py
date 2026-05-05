from __future__ import annotations

from pathlib import Path

from .archive import load_agent
from .defaults import DEFAULT_SELF_MODEL
from .memory import load_preferences
from .models import PreferenceSignal, SelfModel
from .serialization import read_yaml


def compile_system_prompt(root: Path, agent_id: str = "active", *, include_ephemeral: bool = True) -> str:
    agent = load_agent(root, agent_id)
    return compile_system_prompt_for_path(root, agent.path, include_ephemeral=include_ephemeral)


def compile_system_prompt_for_path(root: Path, agent_path: Path, *, include_ephemeral: bool = True) -> str:
    self_model = load_self_model(agent_path)
    task_prompt = (agent_path / "task_prompt.md").read_text(encoding="utf-8")
    sections = [
        _self_model_section(self_model),
        "Behavior prompt:\n" + task_prompt.strip(),
    ]
    if include_ephemeral:
        learned = _learned_preference_section(root)
        if learned:
            sections.append(learned)
    return "\n\n".join(section for section in sections if section.strip()) + "\n"


def load_self_model(agent_path: Path) -> SelfModel:
    data = read_yaml(agent_path / "self_model.yaml", DEFAULT_SELF_MODEL) or DEFAULT_SELF_MODEL
    return SelfModel.model_validate(data)


def _self_model_section(model: SelfModel) -> str:
    return "\n".join(
        [
            "Self model:",
            f"- Public name: {model.name}",
            f"- User: {model.user_name}",
            f"- Default tone with this user: {model.default_tone}",
            f"- Relationship stance: {model.relationship}",
            "- Public identity rules:",
            *[f"  - {rule}" for rule in model.public_identity_rules],
            "- Forbidden outward self-descriptions:",
            *[f"  - {item}" for item in model.forbidden_self_descriptions],
            "- Internal-only details; mention only when the user explicitly asks about internals:",
            *[f"  - {item}" for item in model.internal_only],
        ]
    )


def _learned_preference_section(root: Path, limit: int = 8) -> str:
    preferences = load_preferences(root)[-limit:]
    if not preferences:
        return ""
    return "Recent learned user preferences:\n" + "\n".join(_preference_line(pref) for pref in preferences)


def _preference_line(pref: PreferenceSignal) -> str:
    prefix = "Avoid" if pref.polarity == "negative" else "Prefer"
    line = f"- {prefix} ({pref.target}/{pref.context}, strength={pref.strength:.2f}): {pref.text}"
    if pref.preferred_text:
        line += f"\n  Preferred wording/example: {pref.preferred_text}"
    return line
