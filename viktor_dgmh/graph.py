from __future__ import annotations

from .models import Config, RunState
from .runner import run_one_child


def build_graph(provider, config: Config):
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return None

    graph = StateGraph(RunState)

    def child_node(state: RunState) -> RunState:
        return run_one_child(state, provider, config)

    graph.add_node("select_parent_modify_validate_evaluate_archive_promote", child_node)
    graph.set_entry_point("select_parent_modify_validate_evaluate_archive_promote")
    graph.add_edge("select_parent_modify_validate_evaluate_archive_promote", END)
    return graph.compile()

