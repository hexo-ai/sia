# Configuration

Full reference for SIA's backends, models, and command-line arguments.

## Command-line arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--task` | one of | — | Name of a bundled task: `gpqa`, `lawbench`, `longcot-chess`, `spaceship-titanic` |
| `--task_dir` | one of | — | Path to an external task directory (mutually exclusive with `--task`) |
| `--max_gen` | no | `3` | Number of self-improvement generations |
| `--run_id` | no | `1` | Unique run identifier |
| `--backend` | no | `claude` | Agent backend: `claude` or `openhands` |
| `--meta_model` | no | `haiku` | Model for the meta and feedback agents |
| `--task_model` | no | `claude-haiku-4-5-20251001` | Model for the target agent |

## Backends

### Claude Code (default)

Uses the Claude Agent SDK with Claude models only.

```bash
sia \
  --task gpqa \
  --max_gen 5 \
  --run_id 1 \
  --backend claude \
  --meta_model haiku
```

Supported model shortcuts:

- `haiku` → `claude-haiku-4-5-20251001`
- `sonnet` → `claude-sonnet-4-5-20250929`
- `opus` → `claude-opus-4-5-20251101`

### OpenHands

Uses the OpenHands SDK with support for multiple LLM providers. Use fully-qualified model names.

```bash
sia \
  --task gpqa \
  --max_gen 5 \
  --run_id 2 \
  --backend openhands \
  --meta_model "gemini/gemini-3.1-pro-preview"
```

**Google Gemini:**

```bash
--meta_model "gemini/gemini-3.0-pro"
--meta_model "gemini/gemini-3.1-pro-preview"
```

**OpenAI GPT:**

```bash
--meta_model "openai/gpt-4"
--meta_model "openai/gpt-4-turbo"
```

**Anthropic Claude via OpenHands:**

```bash
--meta_model "anthropic/claude-sonnet-4-5-20250929"
--meta_model "anthropic/claude-opus-4-5-20251101"
```

## API keys

Set the keys for the providers you plan to use:

```bash
# Claude (claude backend, or anthropic/* via openhands)
export ANTHROPIC_API_KEY="..."

# Gemini (openhands backend)
export GOOGLE_API_KEY="..."
# or
export GEMINI_API_KEY="..."

# OpenAI (openhands backend)
export OPENAI_API_KEY="..."

# Generic fallback (openhands backend, if a specific key isn't set)
export LLM_API_KEY="..."
```

## Comparing multiple LLMs on the same task

```bash
# Run 1: Claude via Claude Code (default)
sia --task gpqa --max_gen 3 --run_id 1 \
    --backend claude --meta_model haiku

# Run 2: Gemini via OpenHands
sia --task gpqa --max_gen 3 --run_id 2 \
    --backend openhands --meta_model "gemini/gemini-3.1-pro-preview"

# Run 3: GPT-4 via OpenHands
sia --task gpqa --max_gen 3 --run_id 3 \
    --backend openhands --meta_model "openai/gpt-4"
```

Each run lands in its own `runs/run_{id}/` directory, so they can be compared side by side.

## Notes

- The `claude` backend only accepts the Claude shortcut names (`haiku`, `sonnet`, `opus`). For any other provider, use `--backend openhands`.
- Always use fully-qualified `provider/model` names with the OpenHands backend.
- Make sure the API key matching your chosen model is in the environment before launching.
