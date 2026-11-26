# Changelog

## Unreleased

- TRELLIS.2 shape and texture decoders run their blocks in float16 when the decoder config sets `use_fp16` (every published decoder) instead of float32; norms and the input and output layers stay float32.
- TRELLIS.2 closed-surface reconstruction reuses sparse distances across tile seams and recomputes tiles that exceed the cache budget. The converted model's `mesh.field_cache_bytes` and `mesh.projection_budget_bytes` control the reconstruction memory budgets.
- `convert music --task cover|repaint|complete --source <audio>` on ACE-Step checkpoints: a cover keeps the source's melody, rhythm, chords and structure under a new caption or lyrics (`--cover-strength`), a repaint regenerates `--repaint-start` to `--repaint-end` seconds and keeps the rest, and a complete adds an accompaniment (`--track`) to a single track such as a vocal stem on base checkpoints.

## 0.17.0

- Add an `image_segmentation` task that runs a local SAM 2 checkpoint on a box with optional labeled points and writes a PNG mask. Task admission keeps a 512 MB controller margin.
- `convert separate` splits a mix into a vocal stem and the rest with Mel-Band-RoFormer checkpoints, and `--prompt <text>` separates the named sound with SAM-Audio checkpoints.
- `convert transcribe --words` returns word start and end times, and Whisper decoding resumes after an early end while a second or more of the clip remains.
- `convert generate --image` conditions a Qwen-Image 2.1 render on up to 10 reference stills. Reinstall an existing `mflux` backend to get the update.
- Keyed load-on-request Splash endpoints can bind a named project SSD cache with disk and write caps, and refuse an unsafe cache directory before the worker starts.
- Load-on-request endpoints report prompt and decode stage timing in inference telemetry.
- The `mlx-3d` backend installs MLX 0.32.5. Reinstall an existing `mlx-3d` backend to get the update.
- TRELLIS.2 runs its flow transformer blocks in the dtype the checkpoint config names (bfloat16 in the published models) instead of float32.
- TRELLIS.2 frees the MLX buffer cache before its CPU mesh stages, so 1024 generation completes on a 128 GB Mac.
- TRELLIS.2 and Pixal3D read the cascade token cap and floor from the converted model's `mesh.max_tokens` and `mesh.cascade_floor`.
- Mesh decimation repeats one aggression higher until the face count meets `--max-faces`.
- `convert music` takes `--key` and `--keep-caption` for ACE-Step checkpoints and reports the planner's caption, tempo, key, length, meter and language as `planned`; `convert compose` runs the ACE-Step planner alone; `convert describe` answers a text prompt with no media.
- TRELLIS.2 `convert mesh --bbox X Y Z` fits the sampled shape to the box proportions and the mesh bounds to its sizes.
- Add `mlx_agent.client.convert` for Python tools to run one `convert` verb and receive its JSON `data` or a `ConvertError` carrying the envelope's `code`, `message` and `remediation`.
- `convert scan` reports `local_models` (store, repository, acquisition, task, duplicate groups); `discover --local-status` marks candidates `exact`, `variant` or `absent`, `--exclude-local` leaves local models out before ranking, and task roles (`speech-to-text`, `text-to-speech`, `image-generation`, `video-generation`, `mesh-generation`, `music-generation`) list the Hub by pipeline tag. A projector attached during its base model's conversion is `converted` with that output, and a drafter no port converts is a `companion` instead of `pending`.

## 0.16.0

- Add a Gemini CLI extension with all eight MLX Agent skills, a shared runtime, and reviewable user- and project-scope installs, updates, and removals.
- Add `wire --target openclaw` and `discover --wire --target openclaw` to preview and apply a local model provider in OpenClaw. Wiring an MLX-VLM endpoint now uses port 8083 by default.
- Update the `mlx-vlm` backend to serve text and image requests with the original Inkling 2 Mini checkpoint. Reinstall an existing `mlx-vlm` backend to get the update.
- Bound response header and streaming waits in the JIT gateway and authenticated front door. Client disconnects close the upstream connection, and route changes can be retried before forwarding the request.
- The front door refreshes expired routes before answering and keeps verified routes available during partial discovery failures. Route discovery checks live process identity without polling each model endpoint.
- Keep MCP protocol requests responsive while commands run. Cap execution time, queued calls, and captured output. Cancelling a queued call frees its slot immediately, and completed calls retain their final CLI output.
- Reduce repeated process and model identity checks during serving and task execution. Scope macOS memory accounting to the charged process families and reuse loaded LTX-2 components when weights are retained.
- Harden local state reads and subprocess cleanup, improve stale receipt detection, and recognize macOS system directory aliases when verifying paths.
- Validate the Agy plugin with its bundled logo and omit optional experiment files from per-skill installs on OpenClaw, Hermes, and portable AgentSkills hosts.

## 0.15.0 - 2026-11-21

- Splash JIT workers recheck custom chat templates before each load.
- Workflow activation accepts relative receipt directories.
- The OpenCode plugins require 2.3.2 or later. They register unkeyed chat endpoints and keyed endpoints with readable keys as `mlx-agent/<model name>`, with optional discovery through a shared authenticated front door.
- `convert scan` excludes Ollama manifests configured for remote execution from the local model inventory and skips the Splash compatibility probe when no outputs are eligible.
- Conversion queues use the selected `--receipts-dir` for reconciliation, status and new conversions.
- Splash gateway status reports the configured context limit and, when the loaded worker reports it, the effective limit.
- External memory-budget bindings expire automatically once the original process and its associated group have ended.
- Service discovery excludes missing registered receipt directories from the location limit. Startup can prune a bounded batch of stale registrations and retry ownership discovery.

## 0.14.0 - 2026-11-18

- Add `--json` and `--watch-parent` to `serve front-door`: one result line reports the listening port or the startup failure, and the front door stops when the launching process exits.
- Serve with Splash 1.7.0 or later, and tune it at start or in saved profiles: verify rows, cached-state and KV precision, a server-wide reasoning effort, a thinking budget, the persistent SSD prompt cache and its quota, and a reviewed custom chat template.
- Splash accepts Messages query URLs, keeps ordered system text messages, preserves namespaced Responses tools and streamed tool arguments and forwards reasoning effort; saved profiles carry its memory and context limits into direct and workflow launches, and `serve stop --if-idle` stops a Splash gateway only when idle.
- Add `serve tailnet`: share one API-key-protected chat endpoint over Tailscale Serve after reviewing its HTTPS address; stopping the share keeps the local endpoint running, and remote responses return the selected model ID.
- Refuse a second endpoint for a model that is already served, with the existing endpoint's details in CLI and MCP errors, and add `serve duplication-policy` for models allowed on more than one endpoint. Discovery, duplicate-load errors and workflow roles report one service identity, and workflow members opt into a compatible endpoint with `reuse_existing`.
- Add `serve workflow stage`, which estimates memory for selected roles of an active workflow, and `serve workflow handoff`, which verifies a completed task and retrieves its retained artifacts by task ID.
- Keyed MLX-LM endpoints share verified instruction and tool prefixes, with retained KV bounded and counted in JIT memory admission.
- Add the optional `ltx-video` backend with an `ltx2` port for LTX-2.3 distilled: `convert video` and the served `video_generation` task take `--image` as the first frame and `--audio` with `--audio-start` as driving sound; text-only models refuse them.
- `convert mesh` reuses a warm mesh endpoint; durable media tasks snapshot TRELLIS and LTX conditioning inputs, take a storage threshold and explicit media expiration, and keep provenance after expiry. TRELLIS reconstruction reuses surface fields and deduplicates occupancy faster with byte-identical meshes.

## 0.13.0 - 2026-11-16

- Add `serve front-door`: one authenticated loopback port that lists discovered routes sharing its API key by path-free model name and routes chat, Messages, Responses and embeddings to them. The `mlx-agent` OpenClaw provider plugin discovers its chat models with the gateway key.
- Add Splash as a serving runtime for local GGUF, MLX and Strata profile weights and cached repositories, with authenticated `--jit` serving behind the front door, `strata` and `splash` bench labels and the Splash family of each quantized MLX output in `convert scan`.
- Serve classifiers, text embeddings, speech, music, image, video and 3D models through typed local endpoints that keep weights loaded between jobs or release them after each request, including OpenAI `POST /v1/embeddings` with float or base64 output. `convert speak`, `music`, `generate`, `video` and `describe` run on a live endpoint that already serves the same model, and `task capabilities --model-path` reports model-bound media settings.
- Add the Kev classification port (`jaredpalmer/kev-4b` over Qwen3-4B-Base, 8-bit or unquantized), and serve TypeSafe System One decisions at `POST /v1/systemone` on classification task routes and the front door.
- Protect MLX-LM, vision, Strata, accelerated chat and typed endpoints with API keys, separate control credentials and private Unix or HTTP workers. Add `serve private-worker` and a foreground authenticated MLX service for existing service managers. A live model load refuses when its files or settings no longer match the reviewed serving plan.
- Share memory admission across resident endpoints and tracked agent processes under one budget that includes Metal physical memory; `--memory-budget-gb` caps task jobs. A guard stop releases only the model worker and the endpoint keeps admitting requests, exited owners are not charged, idle task allocator cache is reclaimed before a refused admission is retried, and serving status keeps numeric reasons for memory refusals.
- Add named serving workflows with per-role models and resource profiles, read-only previews, owned activation and memory forecasts that include resident models and pending allocations.
- Managed chat gateways apply serving defaults without overriding explicit client settings and report token usage and response throughput through protected status.
- Convert to FP16 or FP32 and 2, 3, 4, 5, 6 or 8-bit where the converter supports them. Model update preparation fetches pinned sources with companion and support files, requires observed completion and fresh output evidence, and reuses completed downloads instead of starting overlapping ones.
- `convert scan` reports the disk each model occupies with every file counted once, bytes under each root that belong to no model, families of copies of one model across MLX, GGUF and Ollama, vision projector GGUFs with their declared base model, and standalone `.safetensors` and `.ckpt` files; `--compare-assets` compares same-size files by full SHA-256.
- Add the `mlx-task` command and skill to every provider for durable local inference tasks, widen `mlx-advisor` to the whole model lifecycle, and document agent integration and the MCP server.
- TRELLIS.2 `convert mesh` includes the texture stage and writes a closed, outward-facing solid with base color as `COLOR_0` and occlusion, roughness and metallic as `COLOR_1`, takes repeatable `--view` images and a `--guide` GLB, and loads only the components each resolution needs.
- Add the optional `ace-step` backend: `convert music` runs ACE-Step 1.5 checkpoints with their planner, and takes `--bpm`, `--time-signature` and `--reference` for them.
- Add opt-in adaptive denoiser caching for Wan2.1 T2V and TripoSG with feature-memory budgets, refresh limits and reuse statistics, and speed up TripoSG and MiniMax Music3 by reusing conditioning projections, depth prefixes and prompt encodings.
- Fix MLX-VLM tool calls from templates that write `<tool_call><function=` without a newline, such as MiMo-V2.6 and Qwen3.5, size paged Strata MTP expert caches from packed draft-head storage, reap exited serving and worker processes, and update opentelemetry-api to 1.45.2 and the ACE-Step runtime locks.

## 0.12.0 - 2026-11-07

- Add Strata total-memory budgets, live budget changes and saved usage profiles. Choose plain, verified prompt-lookup or MTP decoding; unused draft heads stay unloaded.
- Add image inputs to the Messages and Responses APIs and opt-in image-feature reuse for Qwen2.5-VL and Qwen3-VL dense/MoE models.
- Improve M5 code-edit decoding and prompt processing, reduce idle worker CPU use, and protect active requests during memory-pressure handling and worker reloads.
- Add a coding-edit benchmark and loopback endpoint measurements for repeated prefixes, prefill, concurrency and cold loads. Benchmark exports distinguish end-to-end throughput from native stage rates.
- Add native MLX TripoSG, TRELLIS.2 and Pixal3D mesh generation, including octree surface extraction and camera-aligned glTF output.
- Accept 5-bit and 6-bit conversion where supported, and report converter-specific bit widths and model sizes.
- Rank discovery by downloads, trending, likes or recency, and search repository names.
- Discover MLX Agent services across receipt locations and scope doctor, watch and stop commands to a selected service.
- Fix JIT tool-call identifiers, stream keepalives, end-token trimming, KV-cache admission and readiness deadlines.
- Add `mlx-agent --version` and refresh installation, serving and compatibility documentation.

## 0.11.1 - 2026-11-02

- Fix: `convert start --repo` converts a model downloaded with `intake fetch --revision` from its cached snapshot instead of downloading it again.

## 0.11.0 - 2026-10-30

- Add `serve start --m5-prefill`: an opt-in MLX-LM prompt kernel for affine 4-bit Qwen2 MLP projections on M5 GPUs with macOS 26; other hardware, dependencies, models and inputs keep native MLX, and the plan and receipt record `prefill_kernel`.
- Add `serve start --prompt-lookup`: opt-in MLX-LM decoding for code edits that drafts up to eight tokens from repeated prompt text and verifies them with the model; plain greedy requests run one at a time, other requests keep native generation, and the plan and receipt record `decode_profile`.
- JIT serve gateways answer the Anthropic Messages API (`POST /v1/messages`) and the OpenAI Responses API (`POST /v1/responses`), streamed and unstreamed, with text, system prompts and tool calls translated to the model worker's chat completions; unsupported content and parameters return an `invalid_request_error`.
- The core requires Python 3.11 or later; the `mlx-agent` and `mlx-agent-mcp` launchers exit with a message on older interpreters, including the `python3` bundled with macOS.
- Releases also notify `cavi-ai/plugins`, which points every host in the plugin catalog at the release.
- `convert start --gguf` converts Qwen3.5-family GGUFs (`qwen35`) through a bundled port to MLX-VLM's `qwen3_5_text`, or `qwen3_5` with the matching `mmproj` projector from the same folder; GGUFs holding only a multi-token-prediction block are classified as speculative drafters and refused.
- Update filelock to 4.2.4 and setuptools to 84.2.1 in backend locks.

## 0.10.0 - 2026-10-22

- Fix: credential redaction covers `convert transcribe`, `convert generate` and `convert decide` failure messages, the token after an `Authorization` scheme such as `Bearer`, prefixed Hugging Face, GitHub, OpenAI-style and Slack tokens, and text inside JSON values.
- Fix: the coding role probe checks a model's answer in a separate isolated interpreter with a 5-second limit instead of inside mlx-agent.
- Fix: capture volumes work while other disk images attach or detach. The mount check reads the `hdiutil` listing again instead of failing and leaving the image attached, and a removal preview no longer changes when the capture remounts on another device.
- The `mlx-agent` and `mlx-agent-mcp` launchers cache compiled bytecode under `$XDG_CACHE_HOME/mlx-agent/pycache` (default `~/.cache/mlx-agent/pycache`) instead of recompiling the package on every run, and each command loads only its own modules, so `mlx-agent` starts in less than half the time.
- Inference timeouts report `<verb>_timeout` for every convert command and task kind; `describe`, `speak`, `video` and `music` reported `<verb>_failed`.
- Music tasks reject truncated WAV output, and a changed music output reports `output_artifact_changed` like other media tasks.
- Discovery inspects up to six listed models on Hugging Face at a time instead of one after another.
- CI and release workflows pin every action to a commit and only the release and PyPI jobs receive write permissions; a Coverage / Python 3.12 job runs the suite with the experiment runtime installed, so the lab tests run in CI, and fails when line coverage of the package drops below 84%.
- Configuration transactions record each file in an append-only step journal instead of rewriting the whole receipt, so provider installs no longer slow down as files accumulate; rollback and recovery replay the journal of an unfinished transaction. Provider artifacts are listed with one directory scan, receipt loads and target locks check each folder once instead of once per file, previews are built only up to the preview limit, and provider CLI versions are checked in parallel.
- Fix: a provider CLI newer than its last tested version is detected as `native-visible`, with a `provider_untested` warning in `doctor`, instead of `unsupported`; versions below the minimum stay unsupported.
- Update opentelemetry-api to 1.45.1 in backend locks.

## 0.9.0 - 2026-10-20

- Add `openclaw` and `hermes` providers: the installer places the seven skills, each with its launcher and runtime, in `$OPENCLAW_STATE_DIR/skills` or `<workspace>/skills`, and `$HERMES_HOME/skills` or `<project>/.hermes/skills`.
- Publish the `mlx-agent` package to PyPI with the `mlx-agent` and `mlx-agent-mcp` commands; provider installation from the package reports `plugin_files_unavailable` and points to a checkout.
- Add read-only `mlx_agent_capabilities` discovery for MCP clients and mark invalid CLI arguments as non-retryable.
- Enforce OpenCode extension ownership and clean up partial capture volumes.
- Add policy checks for coding-capture commands.
- Refresh the certifi pins and hashes in the backend dependency locks.
- Update Pillow to 13.0.0 in backend locks and the experiment engine.
- Run the full 27-case MCP battery with MCP Eval 0.8.0 in CI.

## 0.8.0 - 2026-10-19

- Add durable inference tasks (`task start|status|list|result|cancel|verify-model|capabilities`, also through MCP) for classification, image and video generation and understanding, music generation, transcription, and speech synthesis: stable IDs, recent history, cancellation, and verified result artifacts; detached workers keep running after the caller exits and share the compute lease.
- Add optional Strata serving for prepared Qwen3.8-Flash-Next GSQ-RCO GGUF profiles in an isolated runtime, with expert-memory and context-capacity controls checked against memory headroom; speculation is off by default.
- Serving previews describe the weight format and declared quantization.
- Add `serve start --training-receipt` and `--adapter-path`: serve a LoRA adapter on its pinned base; previews, startup, and each load-on-request reload recheck the receipt and the adapter's config and tensor hashes, and adapter tensors count toward the memory estimate.
- Add `lora inspect`: read-only checks of the training executable, the cached base architecture, and both dataset splits.
- Add adapter evaluation prompt suites: up to 16 prompts compared between base and adapter, with per-prompt preferences recorded apart from the generation report.
- Add `lab training-data validate|freeze|inspect` for content-addressed media datasets with caller-selected readiness limits.
- Add study recipes captured from cached Wan 2.1 Diffusers revisions, with weight, asset, runtime, and platform identities rechecked before and after generation.
- Add `serve start --thinking off`: `mlx_lm` serves with the chat template's reasoning switched off; other runtimes record `unsupported` in the reviewed plan.
- `convert scan` outputs report the `backend` that runs them for their task and whether it is installed; cached `mlx-community` snapshots count as MLX outputs even unquantized (Kokoro bf16), and snapshots whose weight files are missing do not.
- `intake resolve` reports `cached` when the Hugging Face cache already holds every file a fetch would download.
- Fix: training refuses splits smaller than one batch and base models whose architecture MLX-LM does not support.
- Fix: `convert scan --mlx-root` inventories MLX models when no GGUF root is configured.
- Fix: `convert scan` groups duplicate MLX model folders and reads Ollama stores under its roots: identical copies as `exact` with reclaimable bytes, other bit widths of one model as `variant`, Ollama models listed under `ollama`.
- Fix: training and model diagnostics honor `HF_HUB_CACHE`, `HF_HOME`, and `XDG_CACHE_HOME`.
- Fix: the MCP server negotiates the protocol version, declares which tools change state, and returns structured results with an output schema beside the text response; a CLI exit is returned as the result's `exit_code` instead of ending the server.

## 0.7.0 - 2026-10-12

- Add local music generation and requantization with checkpoint tokenizer validation.
- Add shared experiment, blind review, workspace adoption, and text LoRA evaluation workflows.
- Add native MLX Wan LoRA training and loading of converted Wan training inputs.
- Add `mlx_conversions` to `intake resolve`: MLX models already quantized from the repository on the Hub (repo, downloads, library), for repositories that are not already MLX.
- Select isolated serving backends from local model metadata.
- Organize installation, model lifecycle, examples, troubleshooting, and contribution documentation into focused guides.
- Remove undeclared Click imports from laboratory regression tests.
- Record GitHub release deployments on the hosted publishing job.

## 0.6.0 - 2026-10-08

- Add `serve start --jit`: the loopback endpoint stays reachable while its local model is unloaded; chat and completion requests load one owned worker from the confirmed local files, concurrent requests share the load, and active streams block unload.
- Add `serve unload` (receipt-fenced, `--expected-pid`) to release JIT weights while the endpoint stays reachable, and separate residency in `serve status`.
- Add JIT memory policy: `--idle-timeout`, `--keep-loaded`, and `--min-headroom-gb` on `serve start` and `serve policy`; `serve policy` updates a running endpoint without a restart; cold loads check fresh OS headroom against estimated weights, runtime allowance, and reserve, and unknown measurements block loading.
- Fix: `convert describe` decodes the answer with the tokenizer's special tokens skipped; moondream3 answers no longer start with `<|md_reserved_4|>`.
- Fix: `convert transcribe` with Whisper vocabularies before large-v3 ends at the end of speech. mlx-audio resolved their missing `<|nospeech|>` to `<|endoftext|>` and suppressed it, so every transcript ran to the token limit with invented text; the runner uses `<|nocaptions|>` instead.

- Add `convert speak`: synthesize one text with a converted text-to-speech model to a new WAV (voice, speed, language), returning audio length, synthesis time, real-time factor, and peak memory (read-only).
- Add `convert describe`: answer one question about an image or a video with a converted vision-language model through its backend, returning token counts, throughput, and timing (read-only).
- Add `convert video`: render one prompt with a converted text-to-video model to a new MP4 (size, frames, fps, steps, seed), returning frame count, duration, generation and load time, seconds per frame, peak memory, and pixel spread (read-only).
- Add the `mlx-video` backend (hash-locked `mlx-video-with-audio` 0.1.37) with a `t2v` port: `intake resolve` and `convert start` take the original Wan2.1 T2V checkpoint (`Wan-AI/Wan2.1-T2V-1.3B`) to an MLX directory at 4 or 8 bits; new task type and backend category `video_generation` (pipeline tag `text-to-video`).
- Add a DeepSeek-V4 DSpark drafter port (`deepseek_v4_dspark`): `convert start --gguf` on llama.cpp's `dflash` drafter GGUF writes an MLX sidecar, tensors named as DeepSeek's checkpoint (`mtp.N` as `stages.N`), FP8-sourced projections affine-quantized, MXFP4 routed experts repacked to MLX `mxfp4` exactly, `config.json` with the DSpark fields and `text_config`. dspark-DeepSeek-V4-Flash-0731-Q8_0 at 4 bits: 4,730 tensors, 10.67 GB; F32, BF16, and FP4 tensors equal DeepSeek-V4-Flash-0731's own safetensors.
- `convert scan` labels speculative-decoding drafters (`dflash`, `eagle3` GGUFs; configs with `dspark_target_layer_ids`) `speculative_draft` (use case `speculative_decoding`), reports `draft` (`port`, `target`, `block_size`), and keys a drafter apart from its target so it is never grouped as the target's duplicate. Drafters no port converts are refused at plan time (`unsupported_draft`).
- Committed provider packages hold one runtime per installed package instead of one per skill: tracked files under `providers/` drop from 1871 to about 200. `providers/claude/` is removed (Claude Code and the installer's `claude` provider use the repository-root plugin files); `providers/codex` and `providers/agy` carry one runtime at the package root, which their skills resolve as `<plugin-root>/scripts/mlx-agent`; `providers/opencode` and `providers/agentskills` carry none.
- `install agentskills` copies the root launcher and runtime into each installed skill, and `install opencode` copies root `src/mlx_agent`; the `cp -R providers/agentskills/...` route is removed from the docs.
- `update` and `uninstall` remove receipt-owned files that the current manifest no longer declares (the per-skill runtime copies of earlier releases), refusing any whose content changed. Transaction changes accept `remove`, receipted with a backup and restored by rollback.
- Scout command and skill text point at the reference packs under the package runtime root (`${CLAUDE_PLUGIN_ROOT}/src/...`, `<plugin-root>/src/...`, `<skill-dir>/src/...`, `<config-dir>/src/...`) instead of a bare `src/...`.
- Fix: the OpenCode `mlx_agent_command` tool accepts all seven capabilities (it accepted only scout, adopt, and wire while bench, doctor, watch, and fleet commands called it).
- Fix: the Codex plugin author is generated from the manifest publisher (`organization_url`) instead of a hardcoded GitHub profile.
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
