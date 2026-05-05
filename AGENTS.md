# Viktor HyperAgent Working Notes

This repository exists to build Viktor as a self-evolving personal HyperAgent.

The goal is not to manually patch prompts whenever the user gives feedback. The goal is to create a runtime and evolution loop where Viktor can observe conversation failures, infer the missing behavior or capability, propose mutations, validate them, judge whether they improve Viktor, and make promoted changes available to the live Slack experience.

## Core Objective

- Viktor should become closer to the user's judgment style over time.
- Judgment style matters more than surface voice: tradeoffs, risk instincts, evidence standards, explanation density, implementation taste, and when to act or stay quiet.
- Writing style still matters, but it should follow from the self-model and observed preferences rather than being hand-patched case by case.
- Capability failures should become structured capability gaps, not fake confidence or prompt-only fixes.

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

Identity, relationship, tone, and internal/external boundary corrections belong in `self_model.yaml`, not scattered across task prompts.

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

- Do not encode one-off conversation incidents as permanent top-level project purpose.
- Keep examples out of this file unless they clarify a general rule.
- Prefer mechanisms that let Viktor learn from future cases over hardcoded fixes for a single case.
- Do not grant new external permissions, mutate accounts, or perform destructive actions without explicit approval.
- Keep tests around the loop: observation, reflection, mutation, validation, judging, promotion, and runtime reload.
