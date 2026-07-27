# Changelog

## Unreleased

## 0.4.0 - 2026-07-27

- Add community bench DB plumbing: `bench run --export` appends a bounded, anonymized result line (repo, chip, runtime, timings; no hostnames, paths, or prompts); `bench aggregate --exports <dir>` dedupes to newest per (repo, chip, runtime) and emits per-(repo, chip) medians; discovery annotates chip-matching candidates with `community_bench` from the bundled aggregate (annotation only, no ranking change).
- Add `mlx-agent fuse`: confirmation-gated LoRA fusion. Validates the adapter directory (`adapter_config.json`), renders the exact `mlx_lm.fuse` argv, and runs detached with a receipt under the same gates as convert/lora. `fuse status` records exits once.
- Add `serve start --launchd`: install a reviewed serve plan as a launchd agent. Renders a deterministic plist (bounded subset, managed label prefix), applies it through the transaction preview/confirm/receipt flow, refuses existing plists, and prints the exact `launchctl bootstrap` command instead of loading it.
- Add `doctor models --prune`: confirmation-gated cleanup of incomplete Hugging Face cache snapshots. The preview lists every candidate directory and byte count and marks the deletion as irreversible; execution requires `--confirm --preview-hash` and removes only cache-owned directories from the reviewed plan.
- Add `mlx-agent lora`: confirmation-gated LoRA training. Validates the dataset (train.jsonl with text or messages per line, bounded) before rendering the exact `mlx_lm.lora` argv; `--confirm --preview-hash` spawns training detached with a receipt. Bounded hyperparameters (iters, batch-size, learning-rate, num-layers); same gates as convert. `lora status` records exits once.
- Add `mlx-agent convert`: confirmation-gated local quantization. Preview renders the exact `mlx_lm.convert` argv and output path; `--confirm --preview-hash` spawns the job detached with a receipt. Gates: source in the HF cache, executable already installed, fresh output path, one job at a time. `convert status` cross-checks receipts against live processes and records exits once.
- Add bundled reference packs: `quantization.md` (quant tradeoff ladder, KV-cache sizing, reasoning-model quant guidance), `model-families.md` (Qwen/Gemma/gpt-oss/Llama template and tool-calling quirks, vision and embedding notes), and `troubleshooting.md` (symptom-first serving playbook). Generated into every provider skill and pointed at from each scout skill.
- Add context-aware fit: discovery extracts bounded architecture facts (layers, KV heads, head dim, max positions) from HF config and attaches an `estimates.kv` block (max context for the host budget, fp16 KV). `discover --context N` tightens `fits` to weights + KV at that context; default weights-only behavior is unchanged.
- Add `mlx-agent watch`: stateful Hugging Face digest. `watch snapshot` records owned inventories (HF cache, runtimes, wired configs) and a full-role discovery reading into one self-owned state file; `watch diff` classifies only owned-relevant changes (new quant of owned, updated tracked repo, gated flip, owned missing).
- Add `mlx-agent fleet`: one-shot per-role router configuration. Renders a bounded LiteLLM router YAML from explicit `--assign role=repo` picks or a completed adopt handoff (`--from-adoption`), with per-role runtime defaults (vision → mlx-vlm, text → mlx_lm) and overrides. Models are checked against local inventories; apply goes through the same preview-confirm-receipt-rollback transaction as wire.
- Promote bench to a full provider capability: `/mlx-bench` on Claude Code, Gemini CLI, and OpenCode, `$mlx-agent:mlx-bench` on Codex, and a portable AgentSkills package, all generated from the manifest with contract parity.
- Add `adopt start --measure`: an optional measure phase between verify and compare that benches verified shortlist candidates (sequential, bounded) and upgrades their evidence to `runtime_measured` while preserving role-probe results. Adoption state schema migrates 1.1/1.2 states to 1.3.
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
