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
$env:CODEX_CLI_SANDBOX="workspace-write"
python -m viktor_dgmh run --generations 1 --children 1
```

The adapter calls `codex exec` in ephemeral mode and captures the final message.
`CODEX_CLI_SANDBOX` accepts `read-only`, `workspace-write`, or
`danger-full-access`. Keep `read-only` for ordinary Q&A; use `workspace-write`
when you want the Slack/Codex agent to make local development edits inside this
project. It is slower than the direct API provider, but avoids copying API keys
into this project.

## Imitation chat loop

The first training surface is local chat. It records prompts, answers, and
lightweight feedback into `memory/`, then turns that into pairwise imitation
cases for future promotion decisions.

```powershell
python -m viktor_dgmh chat --fake --prompt "evaluator가 뭐야?" --feedback "/too-long"
python -m viktor_dgmh memory summarize
python -m viktor_dgmh run --promotion-mode llm_judge --generations 1 --children 1 --fake
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

`llm_judge` promotion is the default. It gives the judge recent transcript,
preferences, imitation cases, prompt diff, and sample active/candidate answers,
then promotes only when the candidate is better with enough confidence and no
safety regression. `imitation_pairwise` remains available as a narrower fallback.

## Conversation reflection self-evolution

Viktor can also create imitation cases without explicit feedback commands. The
reflection loop reviews recent chat transcripts for context misses, overlong
answers, repeated corrections, weak evidence, unsafe instincts, or failure to
act, then stores inferred preferences and pairwise eval cases.

```bash
uv run -m viktor_dgmh reflect --fake
uv run -m viktor_dgmh self-evolve --fake
```

When the Slack app answers, it schedules this reflection/evolution loop in the
background. By default it runs 1 generation x 5 children, guarded by
`auto_evolve_min_chat_events` and `auto_evolve_cooldown_seconds`. Direct
reactions are still useful, but they are only an extra signal; the main loop
learns from the conversation trace itself.

The meta-agent receives an evolution brief containing recent preferences,
imitation cases, and transcript snippets, so child prompts can directly encode
stable style changes such as "use banmal with this user" instead of merely
explaining the issue.

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

The app also listens to messages in public channels where it has been invited.
It uses a conservative router before replying, so it should stay quiet for
ordinary chatter and only join when a message looks like a question or judgment
task. To disable this passive channel mode:

```bash
export SLACK_AUTO_RESPOND_CHANNELS=false
```

If you already installed the app before enabling channel listening, update the
Slack app manifest, reinstall the app, and invite the bot to any channel where
it should observe messages.

When Viktor replies inside a Slack thread, it fetches recent messages from that
same thread and includes them as context for the next answer. If you update the
manifest after this change, reinstall the Slack app so the added history scopes
take effect.

### Slack shell commands

For local development, the Slack app can run explicit shell commands from the
project root. This is disabled by default and should only be enabled for trusted
Slack users.

```bash
export SLACK_ENABLE_SHELL=true
export SLACK_SHELL_ALLOWED_USERS="U12345678"
uv run -m viktor_dgmh slack serve
```

Then DM the bot, mention it, or send this in an invited channel:

```text
!sh pwd
!bash uv run --extra test -m pytest -q
```

Commands run through `bash -lc`, time out after
`slack_shell_timeout_seconds`, truncate output to
`slack_shell_max_output_chars`, and append results to
`memory/shell_commands.jsonl`. `SLACK_SHELL_ALLOW_ANY=true` exists for isolated
local experiments, but it is intentionally not recommended.

## Self-evolving router

The Slack channel router is versioned separately from the answer hyperagent. It
stores its active policy in `router/active.yaml` and logs channel decisions under
`memory/router_observations.jsonl`.

Router feedback reactions:

```text
:eyes: should have responded
:shushing_face: should have stayed silent
```

Inspect and evolve it:

```bash
uv run -m viktor_dgmh router inspect
uv run -m viktor_dgmh router eval --candidate active
uv run -m viktor_dgmh router evolve --children 5
uv run -m viktor_dgmh router promote --candidate <router-id>
```

Evolution is blocked until at least 20 labeled router examples exist.
