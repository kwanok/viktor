"""Pure deterministic helpers for a hyperagent.

This file is validated with AST safety rules before use.
"""


def style_hint(category: str) -> str:
    hints = {
        "architecture": "Give the decision first, then a compact implementation shape.",
        "paper": "Separate claim, evidence, limits, and practical implications.",
        "safety": "Prioritize approval gates and reversible actions.",
        "coding": "Prefer scoped changes, verification, and plain engineering prose.",
        "korean_style": "Use concise Korean with enough structure to scan.",
    }
    return hints.get(category, "Be direct, grounded, and useful.")
