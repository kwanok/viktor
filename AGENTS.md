# Viktor Agent Working Notes

This repository exists for one purpose: build Viktor Agent as a self-evolving agent.

Do not expand the project beyond that. All design choices should serve the self-evolution loop: observe failures, infer missing behavior or capability, propose mutations, validate them, judge whether they improve Viktor, and make promoted changes available to the live runtime.

## Core Objective

- Viktor Agent should improve itself from observed interaction data.
- Current improvement target: become closer to the user's judgment style over time.
- Judgment style means tradeoffs, risk instincts, evidence standards, explanation density, implementation taste, and when to act or stay quiet.
- Surface voice is secondary and should be handled only as part of measurable self-evolution.
- Capability failures should become structured capability gaps, not fake confidence or prompt-only fixes.

## Scope Boundary

- Do not turn Viktor Agent into a broader platform, persona project, product vision, or philosophical claim.
- Do not encode one-off conversation incidents as permanent top-level project purpose.
- Do not manually patch every preference as the main solution; build mechanisms that let Viktor improve from future cases.
- Treat identity, tone, memory, tools, Slack behavior, runtime lifecycle, and judging as implementation surfaces for self-evolution only.

## Agent Roles

- `WorkerAgent` is the outward-facing Viktor runtime that answers and acts within the current permissions.
- `MetaAgent` creates candidate mutations for owned artifacts such as self-model, prompts, policies, and safe helper code.
- `JudgeAgent` evaluates whether a candidate is closer to the intended Viktor behavior and safe enough to promote.
- The runner, validator, approval gates, archive mechanics, and hard safety checks are infrastructure. They should stay stable unless explicitly redesigned.

## Mutable Evolution Targets

These artifacts may evolve through the archive loop:

- `self_model.yaml`
- `task_prompt.md`
- `reflection_policy.yaml`
- `mutator_strategy.yaml`
- `judge_policy.yaml`
- `meta_prompt.md`
- `tool_policy.yaml`
- `memory_policy.yaml`
- `helpers.py`, within validator-approved pure helper rules

Identity, relationship, tone, and internal/external boundary corrections belong in `self_model.yaml`, but only as runtime behavior controls for Viktor Agent.

## Capability Gaps

When Viktor cannot do something the user reasonably expected, record it as a capability gap with evidence, requested capability, failure mode, required changes, restart needs, and status.

Examples of capability gap categories:

- missing Slack action capability
- missing tool permission or scope
- missing runtime lifecycle support
- missing code path
- missing memory or retrieval behavior
- missing evaluation coverage

A capability gap is not automatically a license to implement the capability. The evolution path should still validate scope, safety, permissions, tests, and whether a restart is required.

## Runtime Lifecycle

Self-evolution is incomplete unless promoted changes can reach the live runtime.

- Archive, prompt, policy, and self-model changes should be loadable at message time where possible.
- Code, dependency, Slack scope, manifest, and process-level changes require a supervised restart path.
- Viktor should request restart through the supervisor rather than killing its own process.
- Runtime state such as `memory/`, `runs/`, `runtime/`, generated child archives, and local active pointers are not project doctrine.

## Engineering Boundary

- Keep examples out of this file unless they clarify a general rule.
- Prefer mechanisms that let Viktor learn from future cases over hardcoded fixes for a single case.
- Do not grant new external permissions, mutate accounts, or perform destructive actions without explicit approval.
- Keep tests around the loop: observation, reflection, mutation, validation, judging, promotion, and runtime reload.
