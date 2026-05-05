# Viktor DGM-H Lite

A small CLI-first personal hyperagent lab inspired by DGM-H/Hyperagents.

The first version keeps the runner, validator, and evaluator fixed while allowing
archived hyperagents to mutate their own prompts, policies, and pure helper code.

## Quickstart

```powershell
python -m viktor_dgmh init
python -m viktor_dgmh run --generations 3 --children 5 --fake
python -m viktor_dgmh inspect --agent active
```

For live calls, set:

```powershell
$env:OPENAI_API_KEY="..."
$env:MODEL="gpt-5.5"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
python -m viktor_dgmh run
```

## Codex CLI provider

If you have ChatGPT Pro/Plus connected through Codex CLI, you can let Codex CLI
manage the credentials and use it as an experimental provider:

```powershell
codex login
$env:MODEL_PROVIDER="codex_cli"
$env:CODEX_CLI_MODEL="gpt-5.5"
python -m viktor_dgmh run --generations 1 --children 1
```

The adapter calls `codex exec` in read-only, ephemeral mode and captures the
final message. It is slower than the direct API provider, but avoids copying API
keys into this project.

## Imitation chat loop

The first training surface is local chat. It records prompts, answers, and
lightweight feedback into `memory/`, then turns that into pairwise imitation
cases for future promotion decisions.

```powershell
python -m viktor_dgmh chat --fake --prompt "evaluator가 뭐야?" --feedback "/too-long"
python -m viktor_dgmh memory summarize
python -m viktor_dgmh run --promotion-mode imitation_pairwise --generations 1 --children 1 --fake
```

Feedback commands:

```text
/good
/bad
/too-long
/weak-evidence
/unsafe
/remember <preference>
/rewrite <better answer>
```

`imitation_pairwise` promotion compares active and candidate answers against the
collected imitation cases. A candidate is promoted when it passes validation,
keeps safety above the configured threshold, and wins at least 60% of weighted
pairwise comparisons.

## Slack app

Create a Slack app from `slack_app_manifest.yaml`, install it to your workspace,
enable Socket Mode, and create an app-level token with `connections:write`.

Required environment:

```bash
export SLACK_BOT_TOKEN="xoxb-..."
export SLACK_APP_TOKEN="xapp-..."
export CODEX_CLI_MODEL="gpt-5.5"
```

`MODEL_PROVIDER` defaults to `codex_cli` in `config.yaml`. If you want to use
the OpenAI API directly instead, set `MODEL_PROVIDER=openai` and provide
`OPENAI_API_KEY`.

Run locally:

```bash
uv run -m viktor_dgmh slack serve
```

Use it by DMing the bot or mentioning it in a channel after inviting it. React to
the bot's answer to create feedback:

```text
:+1: good example
:-1: bad example
:scissors: too verbose
:mag: weak evidence
:warning: unsafe
:brain: remember this preference
```
