# Viktor Session Purpose

The current purpose of this project is to build an environment where the Viktor Slack bot can self-evolve in practice, not merely change wording.

Core objective:
- Viktor should observe conversation failures, infer what capability or behavior is missing, improve the relevant artifact or code, validate the change, and make the improved version available to the running Slack experience.

Current architecture direction:
- `WorkerAgent` answers as Viktor.
- `MetaAgent` creates child archive mutations.
- `JudgeAgent` evaluates whether a child should be promoted.
- Archive artifacts such as `self_model.yaml`, `task_prompt.md`, `reflection_policy.yaml`, `mutator_strategy.yaml`, and `judge_policy.yaml` are mutable evolution targets.
- Runner, validator, approval gates, and hard safety checks stay fixed unless explicitly redesigned.

Important current gap:
- The Slack bot currently does not add `:eyes:` reactions to user messages before answering.
- That is a missing Slack capability, not a prompt/style issue.
- Implementing it requires Slack `reactions:write`, manifest updates, code changes in `slack_app.py`, and a running process restart.

Runtime lifecycle principle:
- Self-evolution is incomplete unless promoted archive changes and code/capability changes are reflected in the live Slack process.
- Archive/prompt/policy changes can be loaded at message time.
- Code, Slack scope, manifest, and dependency changes require a supervisor/restart path.
- Viktor should not abruptly kill its own process; it should request restart through a supervisor-controlled mechanism.

Near-term priority:
- Add a daemon/supervisor layer for `slack serve`.
- Add restart/status commands for trusted users.
- Add a restart request file under runtime state.
- Then implement `:eyes:` reaction support as an end-to-end test of capability detection, code change, validation, and restart.

Implementation boundary:
- Do not implement `:eyes:` reaction support directly in the bootstrap step.
- First build capability-gap recording and supervised restart infrastructure.
- `:eyes:` should remain a capability gap that Viktor can later solve through its own evolution path.
