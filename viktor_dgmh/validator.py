from __future__ import annotations

import ast
from pathlib import Path

import yaml

from .models import REQUIRED_AGENT_FILES, Manifest, ValidationIssue, ValidationResult

ALLOWED_IMPORTS = {"math", "statistics", "re", "json", "textwrap", "typing", "collections", "itertools"}
FORBIDDEN_CALLS = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint"}
FORBIDDEN_ATTR_ROOTS = {"os", "sys", "pathlib", "subprocess", "socket", "requests", "urllib", "http", "shutil"}


def validate_agent_dir(agent_path: Path, max_prompt_chars: int = 40_000) -> ValidationResult:
    issues: list[ValidationIssue] = []
    agent_id = agent_path.name
    for filename in REQUIRED_AGENT_FILES:
        if not (agent_path / filename).exists():
            issues.append(ValidationIssue(severity="error", message="Missing required file.", file=filename))

    manifest_path = agent_path / "manifest.yaml"
    if manifest_path.exists():
        try:
            Manifest.model_validate(yaml.safe_load(manifest_path.read_text(encoding="utf-8")))
        except Exception as exc:
            issues.append(ValidationIssue(severity="error", message=f"Invalid manifest: {exc}", file="manifest.yaml"))

    for filename in ("tool_policy.yaml", "memory_policy.yaml"):
        path = agent_path / filename
        if path.exists():
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    issues.append(ValidationIssue(severity="error", message="YAML root must be an object.", file=filename))
            except Exception as exc:
                issues.append(ValidationIssue(severity="error", message=f"Invalid YAML: {exc}", file=filename))

    for filename in ("task_prompt.md", "meta_prompt.md"):
        path = agent_path / filename
        if path.exists() and len(path.read_text(encoding="utf-8")) > max_prompt_chars:
            issues.append(ValidationIssue(severity="error", message="Prompt exceeds maximum size.", file=filename))

    tool_policy_path = agent_path / "tool_policy.yaml"
    if tool_policy_path.exists():
        try:
            policy = yaml.safe_load(tool_policy_path.read_text(encoding="utf-8")) or {}
            allow = policy.get("allow", {})
            if allow.get("shell") is True:
                issues.append(ValidationIssue(severity="error", message="Shell access cannot be enabled in v1.", file="tool_policy.yaml"))
            network = allow.get("network")
            if network not in (False, None, "llm_provider_only"):
                issues.append(ValidationIssue(severity="error", message="Network is limited to llm_provider_only.", file="tool_policy.yaml"))
        except Exception:
            pass

    helper_path = agent_path / "helpers.py"
    if helper_path.exists():
        issues.extend(_validate_helpers(helper_path))

    return ValidationResult(agent_id=agent_id, passed=not any(issue.severity == "error" for issue in issues), issues=issues)


def _validate_helpers(path: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [ValidationIssue(severity="error", message=f"Syntax error: {exc}", file="helpers.py")]

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            else:
                names = [(node.module or "").split(".")[0]]
            for name in names:
                if name and name not in ALLOWED_IMPORTS:
                    issues.append(ValidationIssue(severity="error", message=f"Import not allowed: {name}", file="helpers.py"))

        if isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            root = call_name.split(".")[0] if call_name else ""
            if call_name in FORBIDDEN_CALLS or root in FORBIDDEN_ATTR_ROOTS:
                issues.append(ValidationIssue(severity="error", message=f"Call not allowed: {call_name}", file="helpers.py"))

        if isinstance(node, (ast.With, ast.AsyncWith, ast.Try, ast.ClassDef, ast.Delete, ast.Global, ast.Nonlocal)):
            issues.append(ValidationIssue(severity="error", message=f"Construct not allowed: {type(node).__name__}", file="helpers.py"))

        if isinstance(node, ast.Assign):
            if any(not isinstance(target, (ast.Name, ast.Tuple)) for target in node.targets):
                issues.append(ValidationIssue(severity="error", message="Only simple assignments are allowed.", file="helpers.py"))

    return issues


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""

