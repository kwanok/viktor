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

