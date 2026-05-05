from __future__ import annotations

import argparse
from pathlib import Path

from .archive import init_workspace, load_agent, load_config, persist_score, set_active_agent
from .evaluator import evaluate_agent
from .llm import FakeProvider, OpenAICompatibleProvider
from .models import Config
from .runner import run_evolution
from .validator import validate_agent_dir


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m viktor_dgmh")
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="Create seed archive, config, and benchmark files.")
    init_p.add_argument("--force", action="store_true")

    run_p = sub.add_parser("run", help="Run DGM-H Lite evolution.")
    run_p.add_argument("--generations", type=int, default=3)
    run_p.add_argument("--children", type=int, default=5)
    run_p.add_argument("--fake", action="store_true", help="Use deterministic fake LLM provider.")

    eval_p = sub.add_parser("eval", help="Evaluate one archived hyperagent.")
    eval_p.add_argument("--agent", required=True)
    eval_p.add_argument("--fake", action="store_true")

    inspect_p = sub.add_parser("inspect", help="Inspect one archived hyperagent.")
    inspect_p.add_argument("--agent", required=True)

    promote_p = sub.add_parser("promote", help="Manually set active hyperagent.")
    promote_p.add_argument("--agent", required=True)

    args = parser.parse_args()
    root = Path.cwd().resolve()

    if args.command == "init":
        seed_id = init_workspace(root, force=args.force)
        print(f"Initialized DGM-H Lite workspace. Active hyperagent: {seed_id}")
        return

    config_data = load_config(root)
    config = Config.model_validate(config_data)

    if args.command == "run":
        state = run_evolution(root, args.generations, args.children, use_fake=args.fake, config=config)
        print(f"Run complete: {state.run_id}")
        print(f"Active hyperagent: {load_agent(root, 'active').id}")
        return

    if args.command == "eval":
        agent = load_agent(root, args.agent)
        provider = FakeProvider() if args.fake else OpenAICompatibleProvider.from_env()
        validation = validate_agent_dir(agent.path, config.max_prompt_chars)
        if not validation.passed:
            print(f"Validation failed for {agent.id}")
            for issue in validation.issues:
                print(f"[{issue.severity}] {issue.file or '-'}: {issue.message}")
            return
        score = evaluate_agent(root, agent.path, provider, args.fake)
        persist_score(agent.path, score)
        print(f"Evaluated {agent.id}: total={score.total_score:.4f}, safety={score.safety:.4f}")
        return

    if args.command == "inspect":
        agent = load_agent(root, args.agent)
        validation = validate_agent_dir(agent.path, config.max_prompt_chars)
        print(f"Hyperagent: {agent.id}")
        print(f"Generation: {agent.manifest.generation}")
        print(f"Parent: {agent.parent.parent_id}")
        print(f"Score: total={agent.scores.total_score:.4f}, safety={agent.scores.safety:.4f}")
        print(f"Validation: {'passed' if validation.passed else 'failed'}")
        patch_path = agent.path / "patch.md"
        if patch_path.exists():
            print(patch_path.read_text(encoding="utf-8").strip())
        for issue in validation.issues:
            print(f"[{issue.severity}] {issue.file or '-'}: {issue.message}")
        return

    if args.command == "promote":
        set_active_agent(root, args.agent)
        print(f"Active hyperagent set to {args.agent}")

