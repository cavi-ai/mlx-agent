# MLX Agent overview

mlx-agent discovers, verifies, measures, wires, serves, converts, and trains local MLX-optimized models on Apple Silicon — for your coding agent. It queries the Hugging Face Hub live, matches models to your machine's memory and installed runtimes, proves behavior with deterministic runtime probes, measures real tokens-per-second on your chip, and emits confirmation-gated configuration for the agent you use.

## What it is not

- It never downloads model weights. Discovery and verification read metadata and already-served models only.
- It never installs provider CLIs or model runtimes. You install them; the pack detects them.
- It never mutates configuration without a reviewed preview and an explicit `--confirm --preview-hash`.

## The pipeline

1. **Discover** — `scout` ranks live Hub candidates with real download sizes, license and gating, reasoning detection, and KV-cache-aware fit.
2. **Verify** — `adopt` runs deterministic role probes (coding, reasoning, vision OCR, embedding, tool-use) against your local runtime before recommending.
3. **Measure** — `bench` records TTFT and tok/s on your chip as `runtime_measured` evidence.
4. **Wire** — `wire` applies configuration through a preview-confirm-receipt-rollback transaction.
5. **Operate** — `serve`, `doctor`, `fleet`, and `watch` run the fleet day to day.
6. **Produce** — `convert`, `lora`, and `fuse` quantize, train, and fuse your own models.

## Providers

First-class adapters ship for Claude Code, Codex CLI, Gemini CLI, and OpenCode, plus portable AgentSkills packages. Every provider exposes the four capabilities: `/mlx-scout`, `/mlx-adopt`, `/mlx-wire`, and `/mlx-bench` (`$mlx-agent:` prefix on Codex).
