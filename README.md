# mlx-agent 🍏

> Discover, verify, and **wire** local MLX-optimized models on Apple Silicon — for your coding agent.

<!-- compatibility:begin -->
## Provider support

First-class adapters are included for each provider below. The universal installer supports both user and project scopes.

| Provider | Package | Invoke |
| --- | --- | --- |
| [Claude Code](docs/install/claude.md) | Native plugin | `/mlx-scout`<br>`/mlx-adopt`<br>`/mlx-wire` |
| [Codex CLI](docs/install/codex.md) | Native plugin | `$mlx-agent:mlx-scout`<br>`$mlx-agent:mlx-adopt`<br>`$mlx-agent:mlx-wire` |
| [Gemini CLI](docs/install/gemini.md) | Native extension | `/mlx-scout`<br>`/mlx-adopt`<br>`/mlx-wire` |
| [OpenCode](docs/install/opencode.md) | Native plugin | `/mlx-scout`<br>`/mlx-adopt`<br>`/mlx-wire` |
| [AgentSkills-compatible hosts](docs/install/index.md) | Portable skills | `mlx-scout skill`<br>`mlx-adopt skill`<br>`mlx-wire skill` |
<!-- compatibility:end -->

### Universal installer

```bash
python3 scripts/mlx-agent providers --json
python3 scripts/mlx-agent install gemini --scope user --dry-run --json
# Inspect preview.preview_hash, then explicitly confirm that exact plan:
python3 scripts/mlx-agent install gemini --scope user --confirm --preview-hash <preview-hash> --json
```

The installer is receipt-owned and confirmation-gated: it does not install a host CLI, download model weights, persist secrets, or overwrite unowned files. See the [install overview](docs/install/index.md), [Scout](docs/guides/scout.md), [Adopt](docs/guides/adopt.md), [Wire](docs/guides/wire.md), [security and recovery](docs/security.md), and [v0.1 Claude migration](docs/migrating-from-v0.1.md).

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Platform: Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-black.svg)
![Deps: none](https://img.shields.io/badge/runtime%20deps-none%20(stdlib)-brightgreen.svg)

The MLX model landscape moves weekly. `mlx-agent` queries the HuggingFace Hub live, matches models to your machine's memory and installed runtimes, tells you which are reasoning models (so you don't put one in a fast/cheap slot), and emits the exact config to wire a pick into your agent.

Most tools do **one** of: *run* a model (Ollama, mlx-knife), *calculate* if it fits (VRAM calculators), or *serve* it (mlx_lm). `mlx-agent` is the only headless, agent-native tool that does the whole loop: **discover → verify → wire**.

## What's inside

| Component | What it does |
| --- | --- |
| **`/mlx-scout`** command | Discovery: MLX models on HuggingFace bucketed by role for this host. |
| **`/mlx-adopt`** command | Adoption **workflow** — discover → verify (test-generate) → recommend a per-role routing config. |
| **`/mlx-wire`** command | Render, preview, and apply a runtime/provider configuration transaction (confirmation-gated). |
| **`mlx-scout`** skill | Auto-activates on "which local model?"; wraps the discovery script + runtime reference. |
| **`mlx-advisor`** agent | On-demand expert for picking + wiring a local model for a role. |
| **`scout.py`** | The stdlib-only discovery/wiring core — runs standalone, too. |

## Claude marketplace install

```bash
claude plugin marketplace add cavi-ai/mlx-agent
claude plugin install mlx-agent
```

Then use `/mlx-scout`, `/mlx-adopt`, `/mlx-wire`, or just ask *"what local model should I use for coding?"*

## Provider invocation

- **Codex:** install the native plugin, then invoke `$mlx-agent:mlx-scout`,
  `$mlx-agent:mlx-adopt`, or `$mlx-agent:mlx-wire`. Codex does not support
  custom slash commands, so `/mlx-scout` is incorrect there.
- **Claude Code, Gemini CLI, and OpenCode:** use `/mlx-scout`, `/mlx-adopt`, or
  `/mlx-wire` where their native command adapters are installed.

## Quick look

```console
$ python3 skills/mlx-scout/scripts/scout.py --role reasoning --limit 3

Host: Apple M-series · 128GB · Ollama ✓ · LM Studio ✗

## Reasoning
| model                                   | RAM      | reasoning       | fits | license    |
|-----------------------------------------|----------|-----------------|------|------------|
| mlx-community/gpt-oss-20b-MXFP4-Q8 ⭐    | 12.1GB*  | ⚠ chat_template | ✓    | apache-2.0 |
| mlx-community/Qwen3.6-40B-…-Thinking-8bit ⭐ | 41.5GB* | ⚠ name       | ✓    | apache-2.0 |
| unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit ⭐   | 21.6GB*  | ⚠ name          | ✓    | apache-2.0 |

* = real download size from the HuggingFace tree API (not a guess).
```

## Usage

```bash
python3 skills/mlx-scout/scripts/scout.py                 # all roles
python3 skills/mlx-scout/scripts/scout.py --role coding   # one role
python3 skills/mlx-scout/scripts/scout.py --new           # what changed on HF
python3 skills/mlx-scout/scripts/scout.py --fast          # skip enrichment (faster, name heuristics)
python3 skills/mlx-scout/scripts/scout.py --json          # machine-readable

# emit setup + a ready config block for a chosen model:
python3 skills/mlx-scout/scripts/scout.py --wire <repo> --target mlx_lm|lmstudio|mlx-vlm|ollama|litellm
```

Roles: `general`, `coding`, `reasoning`, `vision`, `embedding`. `--limit N` sets results per role.

## How it works

- **Real sizing** — pulls actual quantized byte size from the HF tree API, not a name guess.
- **Reasoning detection** — reads the model's `chat_template` and tags (catches `reasoning_effort` / `<think>`), falling back to a name heuristic. Reasoning models emit hidden thinking, so `mlx-agent` keeps them out of fast/cheap roles.
- **Quant dedup** — rolls `…-4bit / -8bit / -bf16` up to one logical model and picks the best quant that fits your RAM.
- **License / gated** — surfaces the license and flags gated repos before any external runtime fetch.
- **Verify-before-recommend** — `/mlx-adopt` test-generates a candidate against your local runtime to confirm behavior before wiring it.

## Use anywhere (OpenClaw / Hermes / any agent)

The generated `providers/agentskills/mlx-scout/`, `providers/agentskills/mlx-adopt/`, and `providers/agentskills/mlx-wire/` directories are self-contained [AgentSkills](https://agentskills.io) packages. Copy the complete provider directory you need into an isolated compatible host skills path; each contains its own launcher and runtime. The legacy root `skills/mlx-scout/` is repository-relative compatibility code and is not the portable package.

## Requirements

- macOS on Apple Silicon (for host/RAM/runtime detection; the HF query itself works anywhere)
- Python 3.9+ (standard library only — zero pip installs)
- Optional runtimes it detects & wires: [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai) (MLX), [`mlx_lm`](https://github.com/ml-explore/mlx-lm), [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm) — see [`skills/mlx-scout/references/runtimes.md`](skills/mlx-scout/references/runtimes.md).

## Roadmap

- Tokens/sec-by-chip speed signal in ranking
- Quality/benchmark score beyond download counts
- One-shot fleet setup (wire an entire per-role routing config in one pass)

## Contributing

Issues and PRs welcome. The core is a dependency-free Python package, and `skills/mlx-scout/scripts/scout.py` is its legacy compatibility wrapper — easy to read, easy to extend (add a role, a runtime target, or a better heuristic).

## License

[MIT](LICENSE) © Sasan Sotoodehfar
