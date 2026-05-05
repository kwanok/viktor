from __future__ import annotations

import argparse
from pathlib import Path

from .archive import init_workspace, load_agent, load_config, persist_score, set_active_agent
from .chat import run_chat_once, run_interactive_chat
from .evaluator import evaluate_agent
from .imitation import evaluate_pairwise_imitation
from .llm_judge import evaluate_llm_judge
from .llm import provider_from_config
from .memory import load_imitation_cases, load_preferences
from .models import Config
from .auto_evolve import run_auto_evolve_once
from .reflection import reflect_on_recent_conversation
from .runner import run_evolution
from .router import ensure_router, evaluate_router, evolve_router, load_router_policy, promote_router
from .slack_app import serve_slack_app
from .validator import validate_agent_dir


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m viktor_dgmh")
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="Create seed archive, config, and benchmark files.")
    init_p.add_argument("--force", action="store_true")

    run_p = sub.add_parser("run", help="Run DGM-H Lite evolution.")
    run_p.add_argument("--generations", type=int, default=3)
    run_p.add_argument("--children", type=int, default=5)
    run_p.add_argument("--promotion-mode", choices=["score", "imitation_pairwise", "llm_judge"], default=None)
    run_p.add_argument("--fake", action="store_true", help="Use deterministic fake LLM provider.")

    eval_p = sub.add_parser("eval", help="Evaluate one archived hyperagent.")
    eval_p.add_argument("--agent", required=True)
    eval_p.add_argument("--fake", action="store_true")

    inspect_p = sub.add_parser("inspect", help="Inspect one archived hyperagent.")
    inspect_p.add_argument("--agent", required=True)

    promote_p = sub.add_parser("promote", help="Manually set active hyperagent.")
    promote_p.add_argument("--agent", required=True)

    chat_p = sub.add_parser("chat", help="Chat with the active hyperagent and collect feedback.")
    chat_p.add_argument("--prompt", help="Run one prompt instead of interactive mode.")
    chat_p.add_argument("--feedback", help="Optional feedback command for --prompt.")
    chat_p.add_argument("--fake", action="store_true")

    judge_p = sub.add_parser("judge", help="Run a promotion judge for a candidate.")
    judge_p.add_argument("--candidate", required=True)
    judge_p.add_argument("--against", default="active")
    judge_p.add_argument("--mode", choices=["llm_judge", "imitation_pairwise"], default="llm_judge")
    judge_p.add_argument("--fake", action="store_true")

    memory_p = sub.add_parser("memory", help="Inspect collected imitation memory.")
    memory_p.add_argument("action", choices=["summarize"])

    reflect_p = sub.add_parser("reflect", help="Reflect on recent chat and create imitation cases.")
    reflect_p.add_argument("--fake", action="store_true")
    reflect_p.add_argument("--max-events", type=int, default=40)

    self_evolve_p = sub.add_parser("self-evolve", help="Reflect on conversation memory, then run a small evolution.")
    self_evolve_p.add_argument("--fake", action="store_true")

    slack_p = sub.add_parser("slack", help="Run Slack app integrations.")
    slack_sub = slack_p.add_subparsers(dest="slack_command", required=True)
    slack_serve_p = slack_sub.add_parser("serve", help="Start the Slack Socket Mode app.")
    slack_serve_p.add_argument("--fake", action="store_true")

    router_p = sub.add_parser("router", help="Inspect and evolve the Slack response router.")
    router_sub = router_p.add_subparsers(dest="router_command", required=True)
    router_sub.add_parser("inspect", help="Print active router policy.")
    router_eval_p = router_sub.add_parser("eval", help="Evaluate a router policy on labeled observations.")
    router_eval_p.add_argument("--candidate", default="active")
    router_evolve_p = router_sub.add_parser("evolve", help="Create and evaluate router policy candidates.")
    router_evolve_p.add_argument("--children", type=int, default=5)
    router_evolve_p.add_argument("--min-examples", type=int, default=20)
    router_promote_p = router_sub.add_parser("promote", help="Promote a router candidate if it improves metrics.")
    router_promote_p.add_argument("--candidate", required=True)

    args = parser.parse_args()
    root = Path.cwd().resolve()

    if args.command == "init":
        seed_id = init_workspace(root, force=args.force)
        print(f"Initialized DGM-H Lite workspace. Active hyperagent: {seed_id}")
        return

    config_data = load_config(root)
    config = Config.model_validate(config_data)

    if args.command == "run":
        if args.promotion_mode:
            config.promotion_mode = args.promotion_mode
        state = run_evolution(root, args.generations, args.children, use_fake=args.fake, config=config)
        print(f"Run complete: {state.run_id}")
        print(f"Active hyperagent: {load_agent(root, 'active').id}")
        return

    if args.command == "eval":
        agent = load_agent(root, args.agent)
        provider = provider_from_config(config, root=root, use_fake=args.fake)
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
        return

    if args.command == "chat":
        provider = provider_from_config(config, root=root, use_fake=args.fake)
        if args.prompt:
            session_id, answer = run_chat_once(root, provider, args.prompt, feedback_text=args.feedback)
            print(f"Session: {session_id}")
            print(answer)
        else:
            run_interactive_chat(root, provider)
        return

    if args.command == "judge":
        provider = provider_from_config(config, root=root, use_fake=args.fake)
        if args.mode == "llm_judge":
            result = evaluate_llm_judge(root, args.candidate, provider, active_id=args.against, use_fake=args.fake)
            print(f"Candidate: {result.candidate_id}")
            print(f"Against: {result.active_id}")
            print(f"Winner: {result.winner}")
            print(f"Confidence: {result.confidence:.4f}")
            print(f"Safety regression: {result.safety_regression}")
            print(result.rationale)
            return
        aggregate = evaluate_pairwise_imitation(root, args.candidate, provider, active_id=args.against, use_fake=args.fake)
        if aggregate is None:
            print("No imitation cases found. Use `chat` with feedback first.")
            return
        print(f"Candidate: {aggregate.candidate_id}")
        print(f"Against: {aggregate.active_id}")
        print(f"Win rate: {aggregate.weighted_win_rate:.4f}")
        print(f"Tie rate: {aggregate.weighted_tie_rate:.4f}")
        print(f"Safety regressions: {aggregate.safety_regressions}")
        print(aggregate.rationale)
        return

    if args.command == "memory":
        prefs = load_preferences(root)
        cases = load_imitation_cases(root)
        print(f"Preferences: {len(prefs)}")
        print(f"Imitation cases: {len(cases)}")
        by_kind: dict[str, int] = {}
        for pref in prefs:
            by_kind[pref.kind] = by_kind.get(pref.kind, 0) + 1
        for kind, count in sorted(by_kind.items()):
            print(f"{kind}: {count}")
        return

    if args.command == "reflect":
        provider = provider_from_config(config, root=root, use_fake=args.fake)
        signals, cases = reflect_on_recent_conversation(root, provider, max_events=args.max_events, use_fake=args.fake)
        print(f"Reflection signals: {len(signals)}")
        print(f"Imitation cases: {len(cases)}")
        return

    if args.command == "self-evolve":
        provider = provider_from_config(config, root=root, use_fake=args.fake)
        state = run_auto_evolve_once(root, config, provider, use_fake=args.fake, reason="cli_self_evolve")
        if state is None:
            print("Self-evolve skipped. Check memory/auto_evolve_state.json for the reason.")
            return
        print(f"Self-evolve complete: {state.run_id}")
        print(f"Active hyperagent: {load_agent(root, 'active').id}")
        return

    if args.command == "slack":
        if args.slack_command == "serve":
            try:
                serve_slack_app(root, config, use_fake=args.fake)
            except RuntimeError as exc:
                print(f"Slack app error: {exc}")
                raise SystemExit(1) from exc
        return

    if args.command == "router":
        ensure_router(root)
        if args.router_command == "inspect":
            import yaml

            print(yaml.safe_dump(load_router_policy(root, "active"), sort_keys=False, allow_unicode=True))
            return
        if args.router_command == "eval":
            metrics = evaluate_router(root, args.candidate)
            print(_format_router_metrics(metrics))
            return
        if args.router_command == "evolve":
            try:
                metrics_list = evolve_router(root, children=args.children, min_examples=args.min_examples)
            except RuntimeError as exc:
                print(f"Router evolve blocked: {exc}")
                return
            for metrics in metrics_list:
                print(_format_router_metrics(metrics))
            return
        if args.router_command == "promote":
            promoted = promote_router(root, args.candidate)
            print(f"Router promoted: {promoted}")
            return


def _format_router_metrics(metrics) -> str:
    return (
        f"{metrics.router_id}: examples={metrics.examples} "
        f"precision={metrics.precision:.4f} recall={metrics.recall:.4f} f1={metrics.f1:.4f} "
        f"fpr={metrics.false_positive_rate:.4f} fnr={metrics.false_negative_rate:.4f}"
    )
