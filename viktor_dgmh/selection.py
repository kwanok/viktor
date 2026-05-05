from __future__ import annotations

from pathlib import Path

from .archive import get_active_agent_id, list_agents


def select_parent(root: Path) -> str:
    agents = list_agents(root)
    if not agents:
        raise FileNotFoundError("No archived agents. Run init first.")
    active_id = get_active_agent_id(root)
    valid = [agent for agent in agents if agent.scores.safety >= 0.90]
    if not valid:
        return active_id
    valid.sort(key=lambda agent: (agent.scores.total_score, agent.manifest.generation), reverse=True)
    return valid[0].id

