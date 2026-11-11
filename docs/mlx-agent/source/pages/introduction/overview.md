# MLX Agent overview

mlx-agent helps you choose and operate local models on Apple Silicon for a coding
agent. It combines Hugging Face discovery, local role verification, performance
measurements, configuration transactions, and model lifecycle commands in one
provider-neutral Python core.

## Requirements and boundaries

The core requires Python 3.11+ and uses the standard library. MLX execution
requires Apple Silicon and the appropriate runtime. Optional backend installation
uses Python 3.12 dependency locks; provider CLIs are installed separately.

Discovery, adoption, and benchmarks do not download weights. Explicit
`intake fetch` downloads a reviewed model source; `backend install` installs
declared optional dependencies. Both require a reviewed preview and confirmation.

Wire and Fleet change configuration. Serve launches owned processes. Convert,
LoRA, and Fuse create outputs from local inputs. Those jobs have reviewed starts
and receipts, but are not configuration rollback transactions. Research and
Blueprint write guidance packs without training or scaffolding applications.

## The pipeline

1. **Discover** — `discover` ranks Hub candidates using size, license, gating, reasoning detection, and KV-cache-aware fit estimates.
2. **Verify** — `adopt` runs deterministic role probes (coding, reasoning, vision OCR, embedding, tool-use) against your local runtime before recommending.
3. **Measure** — `bench` records TTFT and tok/s on your chip as `runtime_measured` evidence.
4. **Wire** — `wire` applies configuration through a preview-confirm-receipt-rollback transaction.
5. **Operate** — `serve`, `doctor models`, `fleet`, and `watch` manage local model processes, health, configuration, and Hub changes. Serving covers MLX-LM, MLX-VLM, Strata and Splash, with just-in-time loading, memory budgets and API-key protected endpoints.
6. **Infer** — `task` runs durable classification, embedding, transcription, speech, music, image, video and 3D mesh jobs against local models, with stable IDs, cancellation and verified output files.
7. **Produce** — `intake`, `backend`, `convert`, `lora`, and `fuse` acquire sources, manage dependencies, quantize, train, and fuse local models.

[Your first local model](quickstart.md) covers making weights available and
starting a server before verification. For domain research or project design,
use [Research](../capabilities/research.md) or [Blueprint](../capabilities/blueprint.md).
See [Evidence and sizing](../reference/evidence.md) for probe and fit limits.

## Providers

First-class adapters ship for Claude Code, Codex CLI, Agy, OpenCode, OpenClaw, and Hermes Agent, plus portable AgentSkills packages. Every provider exposes eight capabilities: scout, adopt, wire, bench, doctor, watch, fleet, and task. Claude Code, OpenCode, OpenClaw, and Hermes Agent use `/mlx-*` commands, Codex uses `$mlx-agent:mlx-*`, and Agy and AgentSkills use the named `mlx-*` skills. Claude Code and OpenCode add the `mlx-advisor` agent, and any MCP client can use the bundled MCP server. See [Agent integration](agents.md).
