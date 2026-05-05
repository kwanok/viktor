from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from .archive import copy_parent_to_child, load_agent, next_agent_id, write_child_metadata
from .defaults import DEFAULT_SELF_MODEL
from .llm import ChatProvider
from .memory import load_capability_gaps, load_imitation_cases, load_preferences, load_recent_chat_events
from .models import AggregateScore
from .prompt_compiler import load_self_model
from .serialization import write_json
from .strategy import STRATEGY_FILE_DEFAULTS, load_judge_policy, load_mutator_strategy, load_reflection_policy, strategy_file_text

MUTABLE_FILES = {
    "task_prompt.md",
    "self_model.yaml",
    "reflection_policy.yaml",
    "mutator_strategy.yaml",
    "judge_policy.yaml",
    "meta_prompt.md",
    "tool_policy.yaml",
    "memory_policy.yaml",
    "helpers.py",
}


@dataclass(frozen=True)
class MetaAgent:
    """Evolution role that proposes child archive mutations."""

    root: Path

    def create_child(
        self,
        parent_id: str,
        generation: int,
        child_index: int,
        provider: ChatProvider,
        use_fake: bool = False,
    ) -> tuple[str, Path]:
        parent = load_agent(self.root, parent_id)
        child_id = next_agent_id(self.root, generation, child_index)
        child_path = copy_parent_to_child(self.root, parent_id, child_id)
        proposal = self.request_mutation(parent.path, parent.scores, provider, use_fake)
        summary = proposal.get("mutation_summary", "meta-agent mutation")
        for filename, content in proposal.get("files", {}).items():
            if filename in MUTABLE_FILES:
                (child_path / filename).write_text(str(content), encoding="utf-8")
        write_child_metadata(child_path, child_id, parent_id, generation, summary)
        write_json(child_path / "scores.json", AggregateScore(rationale="Pending evaluation.").model_dump())
        (child_path / "patch.md").write_text(self.patch_summary(parent.path, child_path, summary), encoding="utf-8")
        return child_id, child_path

    def request_mutation(
        self,
        parent_path: Path,
        score: AggregateScore,
        provider: ChatProvider,
        use_fake: bool,
    ) -> dict:
        strategy = load_mutator_strategy(parent_path)
        editable_files = [name for name in strategy.editable_files if name in MUTABLE_FILES]
        if not editable_files:
            editable_files = sorted(MUTABLE_FILES)
        files = {
            name: strategy_file_text(parent_path, name) if name in STRATEGY_FILE_DEFAULTS else self.file_text(parent_path, name)
            for name in editable_files
        }
        evolution_brief = self.build_evolution_brief()
        allowed_files = ", ".join(editable_files)
        prompt = {
            "role": "user",
            "content": (
                "Create a child hyperagent mutation. Return JSON only with keys "
                f"`mutation_summary` and `files`. `files` may contain only: {allowed_files}.\n\n"
                "Use the evolution brief as the primary source for what should change. "
                "Stable user preferences, recurring corrections, and reflection-derived imitation cases "
                "should be converted into concrete edits. Identity, relationship, tone, and internal/external boundary "
                "changes belong in self_model.yaml. Judgment and work behavior changes belong in task_prompt.md. "
                "If reflection misses the right lessons, edit reflection_policy.yaml. If candidate generation keeps choosing "
                "the wrong files or scale, edit mutator_strategy.yaml. If promotion misses regressions or samples the wrong "
                "questions, edit judge_policy.yaml.\n\n"
                f"Active mutator strategy:\n{strategy.model_dump_json(indent=2)}\n\n"
                f"Evolution brief:\n{json.dumps(evolution_brief, ensure_ascii=False, indent=2)}\n\n"
                f"Current aggregate score:\n{score.model_dump_json(indent=2)}\n\n"
                f"Current files:\n{json.dumps(files, ensure_ascii=False, indent=2)}"
            ),
        }
        raw = provider.chat([{"role": "system", "content": files["meta_prompt.md"]}, prompt], response_format="json")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Meta-agent returned invalid JSON: {raw[:500]}") from exc
        if not isinstance(parsed, dict) or not isinstance(parsed.get("files"), dict):
            raise ValueError("Mutation response must be an object with a files object.")
        return parsed

    def file_text(self, path: Path, filename: str) -> str:
        file_path = path / filename
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
        if filename == "self_model.yaml":
            return yaml.safe_dump(DEFAULT_SELF_MODEL, sort_keys=False, allow_unicode=True)
        if filename in MUTABLE_FILES:
            return ""
        raise FileNotFoundError(filename)

    def build_evolution_brief(self) -> dict:
        preferences = load_preferences(self.root)[-12:]
        cases = load_imitation_cases(self.root)[-12:]
        capability_gaps = load_capability_gaps(self.root)[-12:]
        events = load_recent_chat_events(self.root, limit=20)
        transcript = [
            {
                "event_id": event.event_id,
                "session_id": event.session_id,
                "type": event.type,
                "role": event.role,
                "text": event.text,
            }
            for event in events
        ]
        active_path = load_agent(self.root, "active").path
        return {
            "active_self_model": load_self_model(active_path).model_dump(),
            "active_reflection_policy": load_reflection_policy(active_path).model_dump(),
            "active_mutator_strategy": load_mutator_strategy(active_path).model_dump(),
            "active_judge_policy": load_judge_policy(active_path).model_dump(),
            "recent_preferences": [
                {
                    "kind": pref.kind,
                    "polarity": pref.polarity,
                    "strength": pref.strength,
                    "context": pref.context,
                    "target": pref.target,
                    "text": pref.text,
                    "preferred_text": pref.preferred_text,
                }
                for pref in preferences
            ],
            "recent_imitation_cases": [
                {
                    "context": case.context,
                    "prompt": case.prompt,
                    "preference": case.preference,
                    "preferred_text": case.preferred_text,
                    "avoid_text": case.avoid_text,
                    "weight": case.weight,
                }
                for case in cases
            ],
            "recent_capability_gaps": [
                {
                    "gap_id": gap.gap_id,
                    "source": gap.source,
                    "summary": gap.summary,
                    "evidence": gap.evidence,
                    "requested_capability": gap.requested_capability,
                    "failure_mode": gap.failure_mode,
                    "required_changes": gap.required_changes,
                    "requires_restart": gap.requires_restart,
                    "status": gap.status,
                }
                for gap in capability_gaps
            ],
            "recent_transcript": transcript,
        }

    def patch_summary(self, parent_path: Path, child_path: Path, summary: str) -> str:
        changed = []
        for filename in sorted(MUTABLE_FILES):
            old = self.file_text(parent_path, filename)
            new = self.file_text(child_path, filename)
            if old != new:
                changed.append(filename)
        return "# Mutation Summary\n\n" + summary + "\n\nChanged files: " + (", ".join(changed) if changed else "none") + "\n"
