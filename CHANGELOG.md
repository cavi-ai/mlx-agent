# Changelog

## Unreleased

- Add the mflux backend (0.20.0, hash-locked) with an `image_generation` category and a Qwen-Image 2.1 port (`qwen_image_21`): its converter merges an optional PEFT LoRA into the transformer (W + scale·B@A, rounded to the weight dtype), quantizes with mflux, saves mflux's component layout, and writes a root `config.json` (`components`, `quantization`). Diffusers repos resolve by `model_index.json` class (`port_pipelines`); curated `port_recipes` name a repository's pinned base plus its LoRA (abenzerps/Qwen-Image-2.1-Uncensored-GGUF: Qwen/Qwen-Image-2.1@d26bb61 + its LoRA at 1.0, equal bit for bit to its BF16 GGUF on the checked tensors). `intake fetch` downloads a recipe's base snapshot and LoRA; `convert start` reads both from the cache. `intake resolve` reports `recipe`.
- Add `convert generate`: render one prompt with a converted image-generation model to a new PNG (size, steps, seed).
- `convert scan` lists outputs saved as per-component folders when `config.json` names the `components`.
- Add the mlx-embeddings backend (0.1.0, hash-locked) with a `classification` category, and a Laya port (`laya`, convaiinnovations/laya root checkpoint): ModernBERT encoder plus the option-marker decision head, with the shipped temperatures. Repos without a root `config.json` resolve through a port's `port_signatures`; `convert start --model-type` runs a port's own converter (`port_convert`); `port_quantize` takes a `group_size` (Laya: encoder and head linears at group 32).
- Subfolder checkpoints: `org/name/folder` and `tree/<rev>/<folder>` links resolve against that folder's files (config, weights headers, port signature, download size); `intake fetch` downloads only that folder; `convert start --subfolder` converts from the cached folder with any backend's converter. Laya's `multilingual` (mmBERT-base) and `typed-decisions` checkpoints convert this way.
- The convert receipt schema allows `backend` (written by optional-backend conversions) and `source.subfolder`; tests validate real receipts against it.
- Ports can declare the bit widths their conversions keep accurate (`port_bits`); `intake resolve` reports them as `q_bits` and `convert start` refuses others. Laya converts at 8 bits only: at 4 bits the multilingual checkpoint changes 3 of 25 reference answers.
- `intake resolve` reports `download_bytes`, what a snapshot fetch takes; `intake fetch --model-type` downloads only a ported type's files (Laya: 846 MB of the 2.4 GB repo).
- Add `convert decide`: answer typed questions (choice, score, noul) about a state with a converted classification model through its backend (read-only).
- Task labels: `text-classification` and `zero-shot-classification` repos, and models a classification backend implements, are `classification` (use cases: moderation, routing, classification).
- Add `convert transcribe`: transcribe one audio file with a converted speech-to-text model through its backend (read-only).
- `watch diff` no longer reports a gated change when either side is unknown or missing.
- `intake resolve` reports `estimated_output_bytes` (4- and 8-bit) for convertible repos from the safetensors headers; ports declare what they quantize (`port_quantize`). Edge0/Audio8-ASR-Infinite: 3.75 GB at 4 bits, matching the real conversion.
- Add backend model ports: mlx-agent ships an MLX implementation of Edge0/Audio8-ASR-Infinite (`audio8_asr_infinite`: Voxtral Realtime audio tower, Qwen2 decoder with delay conditioning, projector, frame-length embedding, semantic VAD heads, streaming greedy decode with the 30 s rolling window) for the mlx-audio backend. `intake resolve` now reports Audio8 as convertible through mlx-audio; `convert start --backend mlx-audio` installs the port into the backend before converting. Conversion quantizes the decoder only.
- Task labels: a model is vision-language only with a vision config or a vision module (from the mlx-vlm registry's vision files); GGUF main weights are never vision, and GGUF multimodal projectors (`clip`/mmproj) are labelled other.
- Add local-path serving: `serve start --path DIR` serves a local model
  directory (e.g. a converted output outside the Hugging Face cache) through
  the same preview → confirm → receipt flow as `--repo`. The two are mutually
  exclusive; path plans are gated on the directory existing instead of cache
  membership, receipts carry `path` alongside `repo`, and status/stop argv
  matching accepts either.
- Add `intake resolve`: classify a pasted Hugging Face link and report which declared MLX backend (mlx-lm, mlx-vlm, mlx-audio) converts it, or which components have no MLX implementation, without downloading weights.
- `convert scan` items carry a `task` label (type, use cases, source, confidence).
- Add `backend list|install|remove`: optional mlx-vlm and mlx-audio converters in isolated, hash-locked environments (confirmation-gated; removal moves to the Trash).
- Add `convert start --backend` for repo conversions through an installed optional backend.
- Add `intake fetch` and `intake status`: confirmation-gated, receipt-tracked downloads.
- `intake fetch` skips weight formats MLX converters never read (TensorFlow, Flax, ONNX, Core ML, GGUF) in snapshot downloads.
- Add `intake port-analysis`: deterministic component, weight-prefix, and custom-code analysis for architectures no backend implements.
- Add `intake port-plan`: draft a porting plan from that analysis with a locally served model (loopback endpoints only).
- Fix: `convert start --repo` now passes `--quantize`; repo conversions were previously written unquantized.

## 0.5.2 - 2026-08-25

- Replace the discontinued Gemini CLI extension with a first-class Agy plugin package, including isolated native validation, user/project lifecycle coverage, and generated seven-skill bundles.
- Centralize canonical CAVI publisher metadata and project it into generated Claude and Codex metadata without unsupported host fields.
- Rename the former Gemini-only argument parser to shared command parsing, remove its retired file transport, and fix installer secret detection so canonical publisher URLs do not produce false positives.
- Build release documentation artifacts in staging from the exact tagged commit, avoiding impossible self-referential committed provenance.

## 0.5.1 - 2026-07-29

- Rebuild the release documentation artifact in staging with the exact tagged commit so its embedded provenance matches the outer release envelope consumed by CAVI Home.

## 0.5.0 - 2026-07-29

- Add the Release workflow: tagging `v*` validates that every committed copy of the version agrees, runs the full gate set, attaches `mlx-agent-docs-<tag>.tar.gz` to the GitHub Release, and dispatches `cavi-oss-release` to each registered documentation consumer. `scripts/validate_release_version.py` runs the same lockstep check locally, and `scripts/release_envelope.py` renders the artifact manifest and dispatch body from one definition.
- Make the tagged live-runtime health job opt-in behind `MLX_AGENT_LIVE_RUNTIME_HEALTH`: it requires a self-hosted Apple Silicon runner, so tagging a release no longer queues a job that no runner can claim.
- Refresh provider validation evidence for 0.5.0: OpenCode 1.18.9, Codex CLI 0.137.0, and Gemini CLI 0.46.0 smoke suites pass.

- Fix provider roots being judged after symlink resolution: a provider config directory that is a symlink to another location (dotfile checkout, config on another volume) resolved outside the approved user roots and raised, which failed `definitions()` and took down every provider command for every provider. Containment is now judged on the logical path; traversal and absolute escapes are still rejected.
- Install through a symlinked provider directory without configuration: a validated root is resolved once to its real location, so the no-follow descriptor walk only ever sees real directories and previews, receipts, and ownership checks share one spelling. No `XDG_CONFIG_HOME` export is required for a relocated provider directory.
- Add `destination_not_writable`: install, update, and uninstall refuse a root that is a symlink at plan time, naming the symlink and its target, instead of surfacing `[Errno 20]` from the descriptor walk. Doctor still reports such layouts rather than refusing.
- Default installer receipts to `~/.local/state/mlx-agent/installer-receipts` instead of a dotless `~/mlx-agent/installer-receipts`. `MLX_AGENT_CONFIG_ROOT` and `XDG_STATE_HOME` still take precedence in that order.
- Pin host roots in provider, installer, gemini-adapter, and doctor CLI tests so results no longer depend on the developer's own `$HOME`, `~/.config`, or dotfile symlinks.
- Add GGUF sources to convert: `convert scan` inventories local `.gguf` files (bounded stdlib header parse), pairs each with an MLX output by provenance marker, receipt, or name, and groups redundant copies (`exact`) apart from quantization `variant`s without deleting anything; `convert start --gguf PATH` dequantizes with `transformers` then quantizes with `mlx_lm.convert` under the existing preview-confirm-receipt gates, writing an `mlx-converter.json` provenance marker into the output. Adds the bundled `mlx-converter` skill.
- Promote doctor, watch, and fleet to full provider capabilities: `/mlx-doctor`, `/mlx-watch`, and `/mlx-fleet` on Claude Code, Gemini CLI, and OpenCode, `$mlx-agent:`-prefixed on Codex, plus portable AgentSkills packages, all generated from the manifest with contract parity. Process-spawning commands (serve, convert, lora, fuse) remain CLI-only.
- Add release provenance to the documentation artifact manifest: `release.tag` and `release.commit` record the source commit the artifact was built from, matching the bobby-browser consumer pattern.
- Add the immutable documentation artifact: curated docs under `docs/mlx-agent/source`, built by `scripts/build_docs.py` into `docs/mlx-agent/v<version>/` with a `contentSha256` manifest (bobby-browser pattern), enforced by contract tests. See `docs/mlx-agent/CONSUMER.md`.

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
