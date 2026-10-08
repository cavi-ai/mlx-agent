# mlx-agent 🍏

> Discover, verify, and **wire** local MLX-optimized models on Apple Silicon — for your coding agent.

Current package version: **0.6.0**.

<!-- compatibility:begin -->
## Provider support

First-class adapters are included for each provider below. The universal installer supports both user and project scopes.

| Provider | Package | Invoke |
| --- | --- | --- |
| [Claude Code](docs/install/claude.md) | Native plugin | `/mlx-scout`<br>`/mlx-adopt`<br>`/mlx-wire`<br>`/mlx-bench`<br>`/mlx-doctor`<br>`/mlx-watch`<br>`/mlx-fleet` |
| [Codex CLI](docs/install/codex.md) | Native plugin | `$mlx-agent:mlx-scout`<br>`$mlx-agent:mlx-adopt`<br>`$mlx-agent:mlx-wire`<br>`$mlx-agent:mlx-bench`<br>`$mlx-agent:mlx-doctor`<br>`$mlx-agent:mlx-watch`<br>`$mlx-agent:mlx-fleet` |
| [Agy](docs/install/agy.md) | Native plugin | `mlx-scout skill`<br>`mlx-adopt skill`<br>`mlx-wire skill`<br>`mlx-bench skill`<br>`mlx-doctor skill`<br>`mlx-watch skill`<br>`mlx-fleet skill` |
| [OpenCode](docs/install/opencode.md) | Native plugin | `/mlx-scout`<br>`/mlx-adopt`<br>`/mlx-wire`<br>`/mlx-bench`<br>`/mlx-doctor`<br>`/mlx-watch`<br>`/mlx-fleet` |
| [AgentSkills-compatible hosts](docs/install/index.md) | Portable skills | `mlx-scout skill`<br>`mlx-adopt skill`<br>`mlx-wire skill`<br>`mlx-bench skill`<br>`mlx-doctor skill`<br>`mlx-watch skill`<br>`mlx-fleet skill` |
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

Restart Claude Code, then run `/mlx-scout`, `/mlx-adopt`, `/mlx-wire`, `/mlx-bench`, `/mlx-doctor`, `/mlx-watch`, or `/mlx-fleet`.

### Codex CLI

```bash
codex plugin marketplace add cavi-ai/mlx-agent --ref v0.6.0
codex plugin add mlx-agent@mlx-agent
codex plugin list
```

Restart Codex, then invoke `$mlx-agent:mlx-scout`, `$mlx-agent:mlx-adopt`, `$mlx-agent:mlx-wire`, `$mlx-agent:mlx-bench`, `$mlx-agent:mlx-doctor`, `$mlx-agent:mlx-watch`, or `$mlx-agent:mlx-fleet`. Codex does not support custom slash commands.

### Agy

Agy installs the native plugin from its packaged provider directory:

```bash
git clone --depth 1 --branch v0.6.0 https://github.com/cavi-ai/mlx-agent.git
agy plugin validate ./mlx-agent/providers/agy
agy plugin install ./mlx-agent/providers/agy
agy plugin list
```

Restart Agy, use `/skills` to confirm the package loaded, then ask it to use the `mlx-scout`, `mlx-adopt`, `mlx-wire`, `mlx-bench`, `mlx-doctor`, `mlx-watch`, or `mlx-fleet` skill.

### OpenCode

OpenCode uses the confirmation-gated universal installer. Run these commands from a release checkout:

```bash
git clone --depth 1 --branch v0.6.0 https://github.com/cavi-ai/mlx-agent.git
cd mlx-agent
python3 scripts/mlx-agent install opencode --scope user --dry-run --json
# Copy data.preview.preview_hash from the output, then confirm that exact plan:
python3 scripts/mlx-agent install opencode --scope user --confirm --preview-hash <preview-hash> --json
python3 scripts/mlx-agent doctor opencode --scope user --json
```

Restart OpenCode, press `Ctrl+P`, filter for `mlx`, then run `/mlx-scout`, `/mlx-adopt`, `/mlx-wire`, `/mlx-bench`, `/mlx-doctor`, `/mlx-watch`, or `/mlx-fleet`. If OpenCode lives on another volume, set its XDG directories before both installation and launch; see the [OpenCode guide](docs/install/opencode.md).

### AgentSkills-compatible hosts

From a release checkout, install all seven skills into the host's user skills directory with the universal installer. Each installed skill carries its own launcher and runtime, so a skill folder still runs when it is copied alone:

```bash
python3 scripts/mlx-agent install agentskills --scope user --dry-run --json
# Copy data.preview.preview_hash from the output, then confirm that exact plan:
python3 scripts/mlx-agent install agentskills --scope user --confirm --preview-hash <preview-hash> --json
```

For project scope, add `--scope project --project /absolute/project/path`; skills go to `<project>/.agents/skills/`. Restart the host and confirm that `mlx-scout`, `mlx-adopt`, `mlx-wire`, `mlx-bench`, `mlx-doctor`, `mlx-watch`, and `mlx-fleet` appear in its skills list. A skill folder copied by hand from an earlier release is not receipt-owned, so the installer refuses to overwrite it; move existing `mlx-*` folders out of the skills directory before the first `install agentskills`.

### Universal installer and lifecycle

The universal installer supports `claude`, `codex`, `agy`, `opencode`, and `agentskills` in both user and project scopes:

```bash
python3 scripts/mlx-agent providers --json
python3 scripts/mlx-agent install agy --scope user --dry-run --json
# Copy data.preview.preview_hash from the output, then confirm that exact plan:
python3 scripts/mlx-agent install agy --scope user --confirm --preview-hash <preview-hash> --json
python3 scripts/mlx-agent doctor agy --scope user --json
```

Use the same preview-then-confirm sequence for `update` and `uninstall`. Project installs add `--scope project --project /absolute/project/path`. The installer changes only receipt-owned files; it does not download models, persist secrets, overwrite unowned configuration, or modify a provider's marketplace registry.

Hosted documentation is built from `docs/mlx-agent/source` into an immutable artifact (`docs/mlx-agent/CONSUMER.md`). See the [complete install and lifecycle guide](docs/install/index.md), [Scout](docs/guides/scout.md), [Adopt](docs/guides/adopt.md), [Wire](docs/guides/wire.md), [Research](docs/guides/research.md) ([legal](docs/examples/legal-research-pack.md) · [audio](docs/examples/audio-research-pack.md) · [video](docs/examples/video-research-pack.md)), [Blueprint](docs/guides/blueprint.md), [security and recovery](docs/security.md), and [v0.1 Claude migration](docs/migrating-from-v0.1.md).

## What's inside

| Component | What it does |
| --- | --- |
| **Scout** | Discover MLX models on Hugging Face, bucketed by role for this host. |
| **Adopt** | Resume a discover → verify → recommend workflow with durable evidence. |
| **Wire** | Render, preview, and apply a confirmation-gated configuration transaction. |
| **Research** | Build a read-only domain research pack: foundational modalities seed intent; models/adapters/datasets are ranked into markdown + JSON; packs include a justified MLX-native runtime preference (Ollama remains a valid alternate). |
| **Blueprint** | Emit a guidance-only MLX project design pack (quant/train/LoRA/MTX/study notes) under `mlx-blueprints/`. No scaffolding or training. |
| **`mlx-scout`** skill | Auto-activates on "which local model?"; wraps the discovery script + runtime reference. |
| **`mlx-converter`** skill | Auto-activates on "convert this GGUF" / "what have I already converted?"; wraps the GGUF inventory and conversion core. |
| **`mlx-advisor`** agent | On-demand expert for picking + wiring a local model for a role. |
| **`scout.py`** | The stdlib-only discovery/wiring core — runs standalone, too. |
| **Reference packs** | Quant guide, model-family quirks, and a troubleshooting playbook, shipped with the runtime (`src/mlx_agent/resources/references/`). |

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

Research packs (read-only; no download):

```bash
python3 scripts/mlx-agent research --domain "legal contract review" --keyword ocr
python3 scripts/mlx-agent research --domain "meeting notes" --modality audio --facet asr
python3 scripts/mlx-agent research --interview
```

Packs land under `./mlx-research/` as markdown + JSON. See the [research guide](docs/guides/research.md).

Project design packs (guidance only; no scaffolding or training):

```bash
python3 scripts/mlx-agent blueprint --goal "On-device legal OCR assistant" --modality document-vision
python3 scripts/mlx-agent blueprint --goal "Meeting ASR notes" --modality audio --memory-gb 64
```

Design packs land under `./mlx-blueprints/` as markdown + JSON. See the [blueprint guide](docs/guides/blueprint.md).

Performance measurement (read-only; measures only models already served by a running local runtime):

```bash
python3 scripts/mlx-agent bench run --repo mlx-community/Qwen3-32B-4bit --runtime mlx_lm
python3 scripts/mlx-agent bench run --repo qwen3:32b --runtime ollama --runs 5 --json
```

Bench reports TTFT, decode/prefill tok/s, and run spread as `runtime_measured` evidence. `--export <file.jsonl>` appends an anonymized result line you can contribute back; `bench aggregate --exports <dir>` dedupes exports into medians per (repo, chip), and discovery annotates candidates with matching community numbers as `community_bench` (annotation only, no ranking change). It never starts servers and never downloads models. It is also available inside every provider as `/mlx-bench` (`$mlx-agent:mlx-bench` in Codex), and `adopt start --measure` upgrades verified shortlist candidates with measured evidence before ranking.

Model diagnostics (read-only; never deletes or repairs):

```bash
python3 scripts/mlx-agent doctor models                 # inventory + drift + endpoint health
python3 scripts/mlx-agent doctor models --wired-root . --hf-cache ~/.cache/huggingface/hub --json
```

`doctor models --prune` deletes only incomplete cache snapshots, and only after a reviewed candidate list: the preview names every directory and its size, marks the deletion as irreversible, and requires `--confirm --preview-hash`.

`doctor models` inventories the Hugging Face cache and running local runtimes, then reports classified findings: wired configs referencing missing models, wired files changed or deleted since their receipt, two runtimes claiming one port, incomplete cache snapshots, and unreachable wired endpoints.

Serving (confirmation-gated; launches only what you review):

```bash
python3 scripts/mlx-agent serve start --repo mlx-community/Qwen3-32B-4bit --runtime mlx_lm
# inspect the printed plan, then confirm it:
python3 scripts/mlx-agent serve start --repo mlx-community/Qwen3-32B-4bit --runtime mlx_lm --confirm --preview-hash <hash>
python3 scripts/mlx-agent serve status
python3 scripts/mlx-agent serve stop --port 8080
```

Load on request keeps the endpoint reachable without retaining a model worker:

```bash
python3 scripts/mlx-agent serve start --path /absolute/local/model --runtime mlx_lm --port 8080 --jit
# Review the plan and confirm the same arguments with its hash:
python3 scripts/mlx-agent serve start --path /absolute/local/model --runtime mlx_lm --port 8080 --jit --confirm --preview-hash <hash>
python3 scripts/mlx-agent serve status --json
python3 scripts/mlx-agent serve unload --port 8080 --expected-pid <gateway-pid> --json
python3 scripts/mlx-agent serve stop --port 8080
```

`/v1/models` remains available while unloaded. Chat/completion requests start one owned worker using the confirmed local files with Hugging Face offline settings; concurrent cold requests share that load. Unload releases the worker and leaves the gateway reachable; active requests, including streams, block it. Changed model files require a fresh serve plan. Status reports gateway liveness and `model_state` separately. Stop terminates the owned gateway/worker process group. JIT does not support direct `--launchd`; supervise the gateway through the native app or your existing process manager. The gateway serves its selected model only and refuses request-time adapter/draft model overrides.

Optional memory management is part of the confirmed JIT plan:

```bash
python3 scripts/mlx-agent serve start --path /absolute/local/model --runtime mlx_lm --jit --idle-timeout 600 --min-headroom-gb 2
# Confirm with these same arguments and the returned preview hash.
# Change the complete policy on a receipt-owned live gateway without restarting:
python3 scripts/mlx-agent serve policy --port 8080 --expected-pid <gateway-pid> --idle-timeout 600 --min-headroom-gb 2 --json
# Keep weights after use (manual unload remains available):
python3 scripts/mlx-agent serve policy --port 8080 --expected-pid <gateway-pid> --idle-timeout 600 --keep-loaded --min-headroom-gb 2 --json
```

Idle time begins when the last request or stream finishes. `--idle-timeout 0` disables automatic unload; `--keep-loaded` prevents it after use and does not preload weights. A policy command replaces the whole policy: omitting `--keep-loaded` enables idle unload, and omitting `--min-headroom-gb` disables the admission check. These settings persist atomically in the private gateway configuration. Start without policy flags preserves the previous manual-only behavior. Policy flags require JIT.

Before a guarded cold load, fresh OS headroom must cover estimated local weight bytes × 1.10, a 1.5 GB runtime allowance, and the requested reserve (decimal GB). Unknown headroom or weight size fails closed with `memory_unknown`; insufficient space returns `insufficient_headroom`. Both are HTTP 503 responses and leave the endpoint reachable for a later retry. Status includes the policy, dated estimated `memory_check`, and `load_blocked_reason`. The check is not a cross-endpoint reservation: other applications, concurrent cold loads and later context growth can change memory usage.

Fleet routing (one-shot per-role router config, transaction-backed like wire):

```bash
python3 scripts/mlx-agent fleet render --path ./router.yaml --assign coding=mlx-community/Qwen3-32B-4bit --assign vision=mlx-community/Qwen3-VL-8B-Instruct-4bit
python3 scripts/mlx-agent fleet apply --path ./router.yaml --from-adoption ./adopt-state.json
# inspect the diff, then confirm:
python3 scripts/mlx-agent fleet apply --path ./router.yaml --from-adoption ./adopt-state.json --confirm --preview-hash <hash>
```

Watch (stateful owned-model digest):

```bash
python3 scripts/mlx-agent watch snapshot     # record a baseline
python3 scripts/mlx-agent watch diff         # what changed since, for models you own
```

Conversion (confirmation-gated local quantization):

```bash
python3 scripts/mlx-agent convert start --repo meta-llama/Llama-3.1-8B --q-bits 4
# inspect the plan, then confirm:
python3 scripts/mlx-agent convert start --repo meta-llama/Llama-3.1-8B --q-bits 4 --confirm --preview-hash <hash>
python3 scripts/mlx-agent convert status
```

`convert` renders the exact `mlx_lm.convert` argv, requires `--confirm --preview-hash`, then runs the quantization detached with a receipt.

GGUF inventory and conversion:

```bash
python3 scripts/mlx-agent convert scan                       # what is converted, pending, duplicated
python3 scripts/mlx-agent convert scan --pending-only --json
python3 scripts/mlx-agent convert start --gguf ~/models/model-Q4_K_M.gguf --q-bits 4
```

`convert scan` is read-only and stdlib-only: it parses bounded GGUF headers under the configured roots (`--gguf-root`, repeatable), pairs each file with an MLX output by provenance marker, receipt, or name, and groups redundant copies (`exact`, byte-identical or same-quantization) apart from quantization `variant`s. It reports; it never deletes.

```bash
python3 scripts/mlx-agent intake resolve https://huggingface.co/org/name   # what it is, which backend converts it
```

`intake resolve` reads only the model API document and `config.json` (never weights) and returns a verdict: `already_mlx`, `gguf`, `convertible` (an installed backend implements the architecture), `convertible_after_install` (a declared optional backend does), `unsupported` (no backend implements it; reasons and per-component matches say what exists), `blocked` (gated, private, or missing), or `unknown` (Hub unreachable). For `convertible` and `convertible_after_install` verdicts, `estimated_output_bytes` gives the converted size at 4 and 8 bits, computed from the safetensors headers (ranged reads of the header only, following one redirect to Hub storage): matrix weights the backend quantizes cost bits/8 plus an affine scale and bias per group of 64 (or the port's `group_size`), everything else keeps its size; a port's `port_quantize` rule names what its converter quantizes. `convert scan` labels every model with a task type and use cases.

```bash
python3 scripts/mlx-agent backend list
python3 scripts/mlx-agent backend install mlx-audio                      # preview
python3 scripts/mlx-agent backend install mlx-audio --confirm --preview-hash <hash>
python3 scripts/mlx-agent convert start --repo openai/whisper-tiny --backend mlx-audio --q-bits 4
python3 scripts/mlx-agent intake fetch https://huggingface.co/org/name  # preview, then --confirm --preview-hash
python3 scripts/mlx-agent intake status
```

Optional backends (`mlx-vlm`, `mlx-audio`, `mlx-embeddings`, `mflux`, `mlx-video`) install into their own virtual environment under `$XDG_DATA_HOME/mlx-workbench/backends/<id>` from a hash-locked requirements file that ships with mlx-agent; they need the agent to run on Python 3.12 and never touch the main runtime. `backend remove` moves the environment to the Trash. `convert start --backend` runs that backend's converter under the same preview → confirm → receipt gates; convert still never downloads or installs. `intake fetch` downloads a snapshot (or one GGUF file with `--file`) into the Hugging Face cache, or into `--local-dir`, as a receipt-tracked background job; with `--model-type` a ported type's snapshot takes only the port's files. `intake resolve` reports that size as `download_bytes`. A checkpoint in a subfolder is named as `org/name/folder` (or a `tree/<rev>/<folder>` link): intake reads that folder's files, fetch downloads only that folder, and `convert start --repo org/name --subfolder folder` converts it from the cache. A port can limit its conversions to the bit widths it keeps accurate (`port_bits`, reported by intake as `q_bits`).

Conversions use the cached snapshot's local path when present. Pass `convert start --repo org/name --source-path /absolute/downloaded/checkpoint` to preserve an intake download's exact directory, cache and revision. The preview records the source path and fingerprint; confirmation rechecks that identity. Converter workers set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so missing source files fail instead of creating another download.

`convert transcribe --path DIR --audio FILE [--language en]` runs one clip through a converted speech-to-text model in its backend environment and returns the text with timings (read-only; used as a verification canary).

`convert generate --path DIR --prompt TEXT --out /abs/new.png [--width 1024 --height 1024 --steps 40 --seed 42]` renders one prompt with a converted image-generation model in its backend environment and writes only that PNG (used as a verification canary).

`convert speak --path DIR --text TEXT --out /abs/new.wav [--voice NAME --speed 1.0 --lang-code a]` synthesizes one text (1..2000 characters) with a converted text-to-speech model in its backend environment and writes only that 16-bit PCM WAV; it returns `path`, `sample_rate`, `audio_seconds`, `seconds` (synthesis only), `load_seconds`, `real_time_factor`, and `peak_memory_gb` (read-only; used as a verification canary and for timing comparisons).

`convert music --path /abs/local/model --caption TEXT --lyrics '[instrumental]' --out /abs/new.wav [--duration 15 --steps 30 --seed 42 --timeout 3600]` generates music with the isolated audio backend's music loader. MiniMax Music 3 is supported. Lyrics may contain section tags; instrumental prompts use `[instrumental]`. Duration is a maximum request (greater than 0, up to 360 seconds); steps are 1..30. Generation runs offline from existing local weights and writes a new mono or stereo PCM WAV. The result reports audio duration, generation and load time, real-time factor and peak MLX memory; timing does not measure listening quality.

Already-exported quantized MiniMax Music 3 checkpoints can be converted from their local `--source-path`: the music loader dequantizes and requantizes the weights with the model's own layer policy, preserving tokenizer and scheduler assets without downloading or copying the original weight files.

Music generation requires the checkpoint's real local tokenizer (`tokenizer/tokenizer.json` or root `tokenizer.json`). The pinned backend's synthetic tiny-model token fallback is refused; root-level exports use its official prompt encoder.

`convert describe --path DIR --prompt TEXT (--image FILE | --video FILE) [--max-tokens 256 --temperature 0 --fps 1.0 --max-pixels N]` answers one question about a PNG/JPEG/WebP image or an MP4/MOV/M4V clip with a converted vision-language model in its backend environment; it returns `text`, `prompt_tokens`, `generation_tokens`, `prompt_tps`, `generation_tps`, `peak_memory_gb`, `seconds` (generation only), `load_seconds`, and `input`. A processor with native video support gets the clip itself; otherwise the backend's own frame-sampling fallback sends ordered frames, and a model type the backend cannot feed with images or frames is refused as `video_unsupported` (read-only; used as a verification canary).

`convert video --path DIR --prompt TEXT --out /abs/new.mp4 [--width 832 --height 480 --frames 81 --fps N --steps N --seed 42 --timeout 3600]` renders one prompt (1..2000 characters) with a converted text-to-video model in its backend environment and writes only that MP4; width and height are multiples of the model's alignment (16 for Wan2.1), frames requested are 4n+1 (5..241), and `--fps`/`--steps` default to the model's own. It returns `path`, `width`, `height`, `frames` (what the MP4 holds: the Wan2.1 decoder returns 4 frames per latent step, so 5 requested give 8), `fps`, `duration_seconds`, `steps`, `seed`, `seconds` (generation only), `load_seconds`, `seconds_per_frame`, `peak_memory_gb`, and `pixel_std` (std of all pixel values, near 0 for a blank video) (read-only; used as a verification canary and for timing comparisons). `mlx-video` converts the original Wan2.1 T2V checkpoint (`Wan-AI/Wan2.1-T2V-1.3B`, about 17.6 GB to fetch) at 4 or 8 bits and needs no system ffmpeg: the MP4 is encoded by OpenCV.

`convert decide --path DIR --request FILE` answers the typed questions in a JSON request (`{"state": ..., "questions": {"id": {"type": "choice|score|noul", "instructions": ..., "criteria": ...}}}`) with a converted classification model in its backend environment and returns each answer's distribution (read-only; used as a verification canary).

A backend manifest can declare model ports that ship with mlx-agent under `resources/ports/<backend>/`: MLX implementations of architectures the pinned backend lacks (mlx-audio: `audio8_asr_infinite`, Edge0/Audio8-ASR-Infinite; mlx-embeddings: `laya`, convaiinnovations/laya; mflux: `qwen_image_21`, Qwen-Image 2.1, with an optional LoRA merged in; mlx-video: `t2v`, Wan2.1 text-to-video, converted from the original checkpoint layout). A diffusers repository resolves to a port by its `model_index.json` pipeline class, and a curated recipe names a repository whose model is a pinned base checkpoint plus the LoRA it ships. A port can ship its own converter (`port_convert`, selected by `convert start --model-type`), and a repo without a root `config.json` resolves to a port whose `port_signatures` files it carries. Ported types count as supported in `intake resolve` and `backend list`, the convert plan lists them under `ports`, and `convert start --confirm` copies them into the backend's package before the job starts. A port directory carries a digest marker: an up-to-date port is left alone, and a module the backend itself ships under the same name is never replaced.

```bash
python3 scripts/mlx-agent intake port-analysis https://huggingface.co/org/name
python3 scripts/mlx-agent intake port-plan https://huggingface.co/org/name --endpoint http://localhost:8080 --model <served-id>
```

`intake port-analysis` lists, without downloading weights, which components of an unsupported architecture already exist in an MLX backend (with module paths), which weight prefixes and files map to nothing, and the classes in the repository's custom code (parsed, never executed). `intake port-plan` sends that analysis and the custom source to a model you already serve on a loopback port and writes the reply, labeled as a draft with the analysis hash, under `$XDG_STATE_HOME/mlx-workbench/port-plans/`.

`convert start --gguf` dequantizes the GGUF to Hugging Face weights with `transformers`, then quantizes those to MLX with `mlx_lm.convert`, under the same preview-confirm-receipt gates. It needs `torch`, `transformers`, and `gguf` importable by the same interpreter and never installs them. The output carries an `mlx-converter.json` provenance marker naming its source GGUF.

llama.cpp speculative-decoding drafters (`dflash`, `eagle3`) borrow their target's embeddings and output head and run only beside it; `convert scan` labels them `speculative_draft` with a `draft` object (`port`, `target`, `block_size`) and keys them apart from their target. A DeepSeek-V4 DSpark drafter (`dflash` with hyper-connections) converts through the bundled `deepseek_v4_dspark` port (`resources/ports/mlx-lm/`, run with `--port`, needing only `gguf` and `mlx`): tensors keep DeepSeek's checkpoint names with `mtp.N` as `stages.N`, the FP8-sourced projections quantize to the requested bits, the MXFP4 routed experts repack to MLX `mxfp4` without rounding, and `config.json` records `dspark_target_layer_ids`, `dspark_block_size`, `dspark_noise_token_id`, `dspark_markov_rank`, the target's name, and the drafter's DeepSeek-V4 parameters under `text_config`. Other drafters are refused at plan time (`unsupported_draft`).

LoRA training (confirmation-gated, dataset-validated):

```bash
python3 scripts/mlx-agent lora start --repo mlx-community/Qwen3-8B-4bit --data ./my-dataset --iters 1000
# inspect the plan, then confirm:
python3 scripts/mlx-agent lora start --repo mlx-community/Qwen3-8B-4bit --data ./my-dataset --confirm --preview-hash <hash>
python3 scripts/mlx-agent lora status
```

`lora` validates the dataset (`train.jsonl` with `text` or `messages` per line) before rendering the exact `mlx_lm.lora` argv, then trains detached with a receipt under the same gates as convert: cached base model, installed runtime, fresh adapter path, one job at a time. Serve the result with `mlx-agent serve start --adapter-path <out>`, or fuse it into the base weights first:

```bash
python3 scripts/mlx-agent fuse start --repo mlx-community/Qwen3-8B-4bit --adapter ./adapter --confirm --preview-hash <hash>
python3 scripts/mlx-agent fuse status
```

`fuse` validates the adapter (`adapter_config.json` present) and renders the exact `mlx_lm.fuse` argv under the same preview-confirm-receipt gates, producing a standalone fused model. Hard gates: source model already in the Hugging Face cache (never downloads), `mlx_lm` already installed (never installs), fresh output path (never overwrites), one job at a time.

`watch diff` reports only changes relevant to models in your local inventories and wired configs: new quants of an owned base, weight-byte changes on tracked repos, gated flips, and owned models that disappeared. Unlike `discover --new` (which only re-sorts the Hub), watch keeps a baseline snapshot and ignores everything you do not own.

`fleet` renders one bounded LiteLLM router YAML from explicit `--assign role=repo` picks or a completed adopt handoff, defaults vision to `mlx-vlm` and text roles to `mlx_lm`, checks every model against local inventories, and applies through the same preview-confirm-receipt-rollback transaction as wire.

`serve` renders the exact argv and port plan, requires `--confirm --preview-hash`, then spawns the server and writes a receipt. Hard gates: the model must already be in the Hugging Face cache (never downloads), the runtime executable must already be installed (never installs), the port must be free and unclaimed by wired configs, and the bind is loopback-only. `serve stop` only stops processes serve itself started, verified against their receipts. `serve start --launchd` installs the same plan as a launchd agent (preview-confirm, receipt-owned plist, never overwrites) and prints the exact `launchctl bootstrap` command — loading stays with you.

## How it works

- **Real sizing** — pulls actual quantized byte size from the HF tree API, not a name guess.
- **Context-aware fit** — reads model architecture from HF config (layers, KV heads, head dim) and computes KV-cache cost: every candidate carries a `max_context_tokens` estimate for your RAM, and `discover --context N` tightens `fits` to weights + KV at that context.
- **Reasoning detection** — reads the model's `chat_template` and tags (catches `reasoning_effort` / `<think>`), falling back to a name heuristic. Reasoning models emit hidden thinking, so `mlx-agent` keeps them out of fast/cheap roles.
- **Quant dedup** — rolls `…-4bit / -8bit / -bf16` up to one logical model and picks the best quant that fits your RAM.
- **License / gated** — surfaces the license and flags gated repos before any external runtime fetch.
- **Verify-before-recommend** — `/mlx-adopt` test-generates a candidate against your local runtime to confirm behavior before wiring it.
- **Role-fit probes** — deterministic synthetic runtime checks per role: `coding-v1` (AST + sandboxed exec), `reasoning-v1` (exact answer), `vision-v1` (OCR of a synthetic image), `embedding-v1` (cosine ordering). A failed probe rejects the candidate; an unsupported runtime is recorded, not penalized.
- **Measured performance** — `bench run` records real TTFT and tok/s on your chip as `runtime_measured` evidence, which outranks probe-only evidence in comparisons.

For `tool-use`, metadata is not verification. Only a verified, schema-valid synthetic runtime tool call is recommended as tool-use capable. The bounded probe supports Ollama and local OpenAI-compatible LM Studio, `mlx_lm`, and LiteLLM servers; see [Scout evidence](docs/guides/scout.md), [Adopt verification](docs/guides/adopt.md), and the [security boundaries](docs/security.md).

The opt-in release live smoke automatically selects only direct local Ollama, LM Studio, or `mlx_lm` backends. LiteLLM remains supported by the verifier, but is excluded from automatic live selection because a loopback LiteLLM inventory may route to remote or paid backends; smoke it only after separately proving that its route remains local.

## Use anywhere

The generated `providers/agentskills/mlx-*/` directories hold only each skill's `SKILL.md`. `python3 scripts/mlx-agent install agentskills` copies the one runtime from the repository root (`scripts/` and `src/`) into every installed skill, so each skill under `~/.agents/skills/` is a self-contained [AgentSkills](https://agentskills.io) package. Codex and Agy each commit one runtime at the package root (`providers/codex/`, `providers/agy/`), shared by all seven skills. The legacy root `skills/mlx-scout/` is repository-relative compatibility code and is not the portable package.

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
