# Changelog

## Unreleased

- Add `mlx-agent serve`: confirmation-gated launcher for `mlx_lm` and `mlx-vlm` servers. Preview renders the exact argv, port plan, and readiness endpoint; `--confirm --preview-hash` spawns the server and writes a receipt. Hard gates: model present in the Hugging Face cache, runtime executable already installed, port free and unclaimed by wired configs, loopback-only bind. `serve stop` signals only receipt-owned processes after argv verification; `serve status` cross-checks receipts against live processes.
- Add `mlx-agent doctor models`: read-only model diagnostics. Inventories the Hugging Face cache (sizes, revisions, incomplete snapshots) and running loopback runtimes, then reports classified drift findings (missing model, hash mismatch, missing wired file, endpoint conflict) and wired endpoint health. Never deletes, moves, or repairs.
- Add `mlx-agent bench run`: bounded, read-only performance measurement (TTFT, decode/prefill tok/s, run spread) of a model already served by a local loopback runtime; emits `runtime_measured` evidence (`bench-v1`). Never starts servers or downloads models.
- Add deterministic role-fit verification probes: `coding-v1` (AST + sandboxed exec), `reasoning-v1` (exact answer), `vision-v1` (synthetic OCR image via new mlx-vlm runtime client), and `embedding-v1` (cosine ordering via `embed()`). Adopt compare adds a probe bonus and rejects on `role_probe_failed`; unsupported runtimes are recorded, not penalized.
- Add `mlx-agent blueprint`: guidance-only MLX project design packs (quantization, training-loop sketch, LoRA/MTX notes, study materials) under `mlx-blueprints/` as markdown + JSON. No scaffolding, downloads, or training.
- Add justified MLX-native runtime preference to research packs and discovery wiring (`mlx-vlm` / LM Studio / `mlx_lm`) from host inventory and modality/role rules, without changing scoring and without removing Ollama as a valid alternate.
- Add foundational modality layers (`audio`, `video`, `document-vision`) that seed research intents via CLI `--modality`/`--facet`, lexicon detection, or an explicit interview ask; packs include a `## Modality foundations` section. No new discovery roles or runtimes.
- Enrich research packs with ranked PEFT/LoRA adapters and Hub datasets (hybrid list + card scoring via the existing scorer), emit a deterministic dataset blueprint when no datasets match, and write a JSON sidecar beside the markdown pack. Still read-only: no downloads.
- Document verified tool-use recommendations and safety boundaries, and add an opt-in Apple Silicon smoke test that probes the first installed candidate on supported loopback runtimes.
- Add `mlx-agent research`: read-only domain research packs. An interview builds a validated domain intent; a transparent scoring core ranks models from Hugging Face metadata and bounded model-card text; results are written as project-local markdown under `mlx-research/`. No verification, wiring, or downloads.

## 0.3.0 - 2026-07-20

- Route OpenCode user-scope artifacts through `XDG_CONFIG_HOME` while preserving native `HOME`.
- Route user-scope installer receipts through `XDG_STATE_HOME` unless `MLX_AGENT_CONFIG_ROOT` is explicitly set.
- Record OpenCode 1.18.3 native command discovery and the isolated install/uninstall lifecycle.
- Document complete provider invocation, installation, update, verification, and recovery paths.
- Retain confirmation-gated, receipt-owned mutations and provider-specific command syntax.

## 0.2.0 - 2026-07-17

- Added the provider-neutral Scout, Adopt, and Wire core.
- Added native Claude Code, Codex CLI, Gemini CLI, and OpenCode adapters plus portable AgentSkills packages.
- Added deterministic generation, compatibility contracts, transactional installation, and recovery evidence.

## 0.1.0

- Initial Claude marketplace release and legacy Scout workflow.
