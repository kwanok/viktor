from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .archive import get_active_agent_id, load_agent, persist_score, set_active_agent
from .evaluator import evaluate_agent
from .llm import ChatProvider, FakeProvider, OpenAICompatibleProvider
from .models import Config, RunState
from .mutator import create_child
from .paths import runs_dir
from .selection import select_parent
from .serialization import append_jsonl, write_json
from .validator import validate_agent_dir


def run_evolution(
    root: Path,
    generations: int = 3,
    children: int = 5,
    provider: ChatProvider | None = None,
    use_fake: bool = False,
    config: Config | None = None,
) -> RunState:
    config = config or Config()
    provider = provider or (FakeProvider() if use_fake else OpenAICompatibleProvider.from_env())
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    state = RunState(
        root=root,
        run_id=run_id,
        generations=generations,
        children=children,
        use_fake=use_fake,
        active_agent_id=get_active_agent_id(root),
    )
    graph = _build_optional_graph(provider, config)
    for generation in range(1, generations + 1):
        state.current_generation = generation
        for child_index in range(1, children + 1):
            state.current_child = child_index
            if graph is None:
                state = run_one_child(state, provider, config)
            else:
                state = RunState.model_validate(graph.invoke(state))
    _write_run_summary(state)
    return state


def run_one_child(state: RunState, provider: ChatProvider, config: Config) -> RunState:
    root = state.root
    parent_id = select_parent(root)
    state.selected_parent_id = parent_id
    _event(state, "select_parent", {"parent_id": parent_id})

    try:
        child_id, child_path = create_child(
            root,
            parent_id,
            state.current_generation,
            state.current_child,
            provider,
            state.use_fake,
        )
        state.candidate_id = child_id
        state.candidate_path = child_path
        _event(state, "modify_child", {"child_id": child_id})

        validation = validate_agent_dir(child_path, config.max_prompt_chars)
        state.validation = validation
        _event(state, "validate_child", validation.model_dump())
        if not validation.passed:
            return state

        score = evaluate_agent(root, child_path, provider, state.use_fake)
        state.candidate_score = score
        persist_score(child_path, score)
        _event(state, "evaluate_child", {"child_id": child_id, "score": score.total_score, "safety": score.safety})

        promoted = maybe_promote(root, child_id, score, config)
        state.promoted = promoted
        if promoted:
            state.active_agent_id = child_id
        _event(state, "maybe_promote", {"child_id": child_id, "promoted": promoted})
    except Exception as exc:
        _event(state, "error", {"type": type(exc).__name__, "message": str(exc)})
    return state


def maybe_promote(root: Path, candidate_id: str, candidate_score, config: Config) -> bool:
    active = load_agent(root, "active")
    if candidate_score.safety < config.promotion_min_safety:
        return False
    if candidate_score.total_score >= active.scores.total_score + config.promotion_delta:
        set_active_agent(root, candidate_id)
        return True
    return False


def _event(state: RunState, event: str, payload: dict) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": state.run_id,
        "event": event,
        "generation": state.current_generation,
        "child": state.current_child,
        "payload": payload,
    }
    state.events.append(record)
    append_jsonl(runs_dir(state.root) / f"{state.run_id}.jsonl", record)


def _write_run_summary(state: RunState) -> None:
    promoted = [event for event in state.events if event["event"] == "maybe_promote" and event["payload"].get("promoted")]
    errors = [event for event in state.events if event["event"] == "error"]
    summary = {
        "run_id": state.run_id,
        "generations": state.generations,
        "children": state.children,
        "active_agent_id": get_active_agent_id(state.root),
        "promotions": len(promoted),
        "errors": len(errors),
        "events": len(state.events),
    }
    write_json(runs_dir(state.root) / f"{state.run_id}_summary.json", summary)


def _build_optional_graph(provider: ChatProvider, config: Config):
    try:
        from .graph import build_graph
    except Exception:
        return None
    return build_graph(provider, config)
