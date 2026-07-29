# Installation

Requires macOS on Apple Silicon, Python 3.9+ (standard library only), and the coding-agent host of your choice. Optional runtimes (Ollama, LM Studio, `mlx_lm`, `mlx-vlm`, LiteLLM) are detected, never installed for you.

## Claude Code

```bash
claude plugin marketplace add cavi-ai/mlx-agent
claude plugin install mlx-agent@mlx-agent
```

Restart Claude Code, then run `/mlx-scout`, `/mlx-adopt`, `/mlx-wire`, `/mlx-bench`, `/mlx-doctor`, `/mlx-watch`, or `/mlx-fleet`.

## Codex CLI

```bash
codex plugin marketplace add cavi-ai/mlx-agent --ref v0.5.0
codex plugin add mlx-agent@mlx-agent
```

Restart Codex, then invoke `$mlx-agent:mlx-scout` and the other skills. Codex does not support custom slash commands.

## Gemini CLI

```bash
git clone --depth 1 --branch v0.5.0 https://github.com/cavi-ai/mlx-agent.git
gemini extensions install ./mlx-agent/providers/gemini
```

Restart Gemini CLI, then run `/mlx-scout` and the other commands.

## OpenCode

```bash
git clone --depth 1 --branch v0.5.0 https://github.com/cavi-ai/mlx-agent.git
cd mlx-agent
python3 scripts/mlx-agent install opencode --scope user --dry-run --json
python3 scripts/mlx-agent install opencode --scope user --confirm --preview-hash <preview-hash> --json
python3 scripts/mlx-agent doctor opencode --scope user --json
```

Restart OpenCode, press `Ctrl+P`, filter for `mlx`.

## Universal installer

The same preview-then-confirm installer covers `claude`, `codex`, `gemini`, and `opencode` in user or project scope:

```bash
python3 scripts/mlx-agent providers --json
python3 scripts/mlx-agent install gemini --scope user --dry-run --json
python3 scripts/mlx-agent install gemini --scope user --confirm --preview-hash <preview-hash> --json
```

The installer changes only receipt-owned files: no model downloads, no persisted secrets, no edits to unowned configuration.
