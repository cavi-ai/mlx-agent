# mlx-agent 🍏

> Discover, verify, and **wire** local MLX-optimized models on Apple Silicon — for your coding agent.

Current package version: **0.3.0**.

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

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Platform: Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-black.svg)
![Deps: none](https://img.shields.io/badge/runtime%20deps-none%20(stdlib)-brightgreen.svg)

The MLX model landscape moves weekly. `mlx-agent` queries the HuggingFace Hub live, matches models to your machine's memory and installed runtimes, tells you which are reasoning models (so you don't put one in a fast/cheap slot), and emits the exact config to wire a pick into your agent.

Most tools focus on running, sizing, or serving models. `mlx-agent` connects those concerns into an agent-oriented **discover → verify → wire** workflow.

## Install

Install the package for the coding-agent host you use. The host CLI must already be installed; `mlx-agent` does not install provider CLIs or model runtimes.

### Claude Code

```bash
claude plugin marketplace add cavi-ai/mlx-agent
claude plugin install mlx-agent@mlx-agent
claude plugin list
```

Restart Claude Code, then run `/mlx-scout`, `/mlx-adopt`, or `/mlx-wire`.

### Codex CLI

```bash
codex plugin marketplace add cavi-ai/mlx-agent --ref v0.3.0
codex plugin add mlx-agent@mlx-agent
codex plugin list
```

Restart Codex, then invoke `$mlx-agent:mlx-scout`, `$mlx-agent:mlx-adopt`, or `$mlx-agent:mlx-wire`. Codex does not support custom slash commands.

### Gemini CLI

Gemini installs this extension from its packaged provider directory:

```bash
git clone --depth 1 --branch v0.3.0 https://github.com/cavi-ai/mlx-agent.git
gemini extensions install ./mlx-agent/providers/gemini
gemini extensions list
```

Restart Gemini CLI, then run `/mlx-scout`, `/mlx-adopt`, or `/mlx-wire`.

### OpenCode

OpenCode uses the confirmation-gated universal installer. Run these commands from a release checkout:

```bash
git clone --depth 1 --branch v0.3.0 https://github.com/cavi-ai/mlx-agent.git
cd mlx-agent
python3 scripts/mlx-agent install opencode --scope user --dry-run --json
# Copy data.preview.preview_hash from the output, then confirm that exact plan:
python3 scripts/mlx-agent install opencode --scope user --confirm --preview-hash <preview-hash> --json
python3 scripts/mlx-agent doctor opencode --scope user --json
```

Restart OpenCode, press `Ctrl+P`, filter for `mlx`, then run `/mlx-scout`, `/mlx-adopt`, or `/mlx-wire`. If OpenCode lives on another volume, set its XDG directories before both installation and launch; see the [OpenCode guide](docs/install/opencode.md).

### AgentSkills-compatible hosts

From a release checkout, copy all three self-contained packages into the host's user skills directory:

```bash
mkdir -p ~/.agents/skills
cp -R providers/agentskills/mlx-scout providers/agentskills/mlx-adopt providers/agentskills/mlx-wire ~/.agents/skills/
```

For project scope, copy them to `<project>/.agents/skills/` instead. Restart the host and confirm that `mlx-scout`, `mlx-adopt`, and `mlx-wire` appear in its skills list.

### Universal installer and lifecycle

The universal installer supports `claude`, `codex`, `gemini`, and `opencode` in both user and project scopes:

```bash
python3 scripts/mlx-agent providers --json
python3 scripts/mlx-agent install gemini --scope user --dry-run --json
# Copy data.preview.preview_hash from the output, then confirm that exact plan:
python3 scripts/mlx-agent install gemini --scope user --confirm --preview-hash <preview-hash> --json
python3 scripts/mlx-agent doctor gemini --scope user --json
```

Use the same preview-then-confirm sequence for `update` and `uninstall`. Project installs add `--scope project --project /absolute/project/path`. The installer changes only receipt-owned files; it does not download models, persist secrets, overwrite unowned configuration, or modify a provider's marketplace registry.

See the [complete install and lifecycle guide](docs/install/index.md), [Scout](docs/guides/scout.md), [Adopt](docs/guides/adopt.md), [Wire](docs/guides/wire.md), [Research](docs/guides/research.md), [security and recovery](docs/security.md), and [v0.1 Claude migration](docs/migrating-from-v0.1.md).

## What's inside

| Component | What it does |
| --- | --- |
| **Scout** | Discover MLX models on Hugging Face, bucketed by role for this host. |
| **Adopt** | Resume a discover → verify → recommend workflow with durable evidence. |
| **Wire** | Render, preview, and apply a confirmation-gated configuration transaction. |
| **Research** | Build a read-only domain research pack: an interview scores and ranks models from Hugging Face metadata and model cards into project-local markdown. |
| **`mlx-scout`** skill | Auto-activates on "which local model?"; wraps the discovery script + runtime reference. |
| **`mlx-advisor`** agent | On-demand expert for picking + wiring a local model for a role. |
| **`scout.py`** | The stdlib-only discovery/wiring core — runs standalone, too. |

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

Roles: `general`, `coding`, `reasoning`, `vision`, `embedding`, and `tool-use`. A model retains its primary role and can also have `tool-use` membership. `--limit N` sets results per role.

## How it works

- **Real sizing** — pulls actual quantized byte size from the HF tree API, not a name guess.
- **Reasoning detection** — reads the model's `chat_template` and tags (catches `reasoning_effort` / `<think>`), falling back to a name heuristic. Reasoning models emit hidden thinking, so `mlx-agent` keeps them out of fast/cheap roles.
- **Quant dedup** — rolls `…-4bit / -8bit / -bf16` up to one logical model and picks the best quant that fits your RAM.
- **License / gated** — surfaces the license and flags gated repos before any external runtime fetch.
- **Verify-before-recommend** — `/mlx-adopt` test-generates a candidate against your local runtime to confirm behavior before wiring it.

For `tool-use`, metadata is not verification. Only a verified, schema-valid synthetic runtime tool call is recommended as tool-use capable. The bounded probe supports Ollama and local OpenAI-compatible LM Studio, `mlx_lm`, and LiteLLM servers; see [Scout evidence](docs/guides/scout.md), [Adopt verification](docs/guides/adopt.md), and the [security boundaries](docs/security.md).

The opt-in release live smoke automatically selects only direct local Ollama, LM Studio, or `mlx_lm` backends. LiteLLM remains supported by the verifier, but is excluded from automatic live selection because a loopback LiteLLM inventory may route to remote or paid backends; smoke it only after separately proving that its route remains local.

## Use anywhere

The generated `providers/agentskills/mlx-scout/`, `providers/agentskills/mlx-adopt/`, and `providers/agentskills/mlx-wire/` directories are self-contained [AgentSkills](https://agentskills.io) packages. Copy the complete provider directory you need into an isolated compatible host skills path; each contains its own launcher and runtime. The legacy root `skills/mlx-scout/` is repository-relative compatibility code and is not the portable package.

## Requirements

- macOS on Apple Silicon (for host/RAM/runtime detection; the HF query itself works anywhere)
- Python 3.9+ (standard library only — zero pip installs)
- Optional runtimes it detects & wires: [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai) (MLX), [`mlx_lm`](https://github.com/ml-explore/mlx-lm), [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm), and [LiteLLM](https://www.litellm.ai/) — see [`skills/mlx-scout/references/runtimes.md`](skills/mlx-scout/references/runtimes.md).

## Roadmap

- Tokens/sec-by-chip speed signal in ranking
- Quality/benchmark score beyond download counts
- One-shot fleet setup (wire an entire per-role routing config in one pass)

## Contributing

Issues and PRs welcome. The core is a dependency-free Python package, and `skills/mlx-scout/scripts/scout.py` is its legacy compatibility wrapper — easy to read, easy to extend (add a role, a runtime target, or a better heuristic).

## License

[MIT](LICENSE) © Sasan Sotoodehfar
