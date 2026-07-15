---
name: mlx-scout
description: "Discover MLX-optimized models on HuggingFace suited to this Apple Silicon host, routed by role (general, coding, reasoning, vision, embedding)."
homepage: https://github.com/sasan1200/mlx-agent
license: MIT
metadata:
  openclaw:
    emoji: "🍏"
    requires:
      bins: ["python3"]
---

# mlx-scout

Find local MLX models worth running on this Mac. Queries the HuggingFace Hub for `mlx`-tagged models, detects host RAM + runtimes (Ollama / LM Studio), buckets by role, and — for the top candidates — pulls **real download size**, **reasoning detection**, and **license/gated** status, deduping quant variants.

## Use when

- Picking a local model for a role: fast/cheap general, capable general, coding, reasoning, vision, embeddings.
- Checking what is new on HF for Apple Silicon (`--new`).
- Emitting a ready runtime config for a chosen model (`--wire`).
- Refreshing local model choices on a schedule (see cron below).

## Run

- All roles: `python3 scripts/scout.py`
- One role: `python3 scripts/scout.py --role coding`
- What changed recently: `python3 scripts/scout.py --new`
- Fast (skip HF enrichment, name heuristics only): `python3 scripts/scout.py --fast`
- Machine-readable: `python3 scripts/scout.py --json`
- Emit setup + config for a model: `python3 scripts/scout.py --wire <repo> --target ollama|lmstudio|mlx_lm|mlx-vlm|litellm [--port N]`

Stdlib-only, no install. `--limit N` sets results per role. Enrichment makes ~2 HF calls per shown model; use `--fast` to skip it.

## Reading the output

- ⭐ marks a reputable publisher (mlx-community, lmstudio-community, unsloth, …). Quant variants of the same base model are deduped to the best-fitting one.
- **RAM** with `*` = real download size from the HuggingFace tree API; without `*` = estimated from params/name. `fits` = weights under ~80% of host RAM — add KV-cache headroom (~1–4GB) for long context.
- **`reasoning ⚠`** shows its source: `chat_template` (strong — the model's template carries `reasoning_effort`/`<think>`), `tags`, or `name` (weakest). Reasoning models emit hidden thinking, so never route one to a fast/cheap role. Even on a "no", confirm with a quick generate — a reasoner fills a short `num_predict` with a thinking preamble and returns empty `content`.
- **license** column shows the license and 🔒 if the repo is gated (you must accept terms before pulling).

## Wiring a pick into your agent

Use `--wire <repo> --target <t>` to emit the exact pull/serve command **and** a ready config block:

- `ollama` → curated tags / HF GGUF, ref `ollama/<tag>` (Ollama's MLX engine, Macs ≥32GB)
- `lmstudio` → `lms get`/`lms server start`, ref `lmstudio/<model>`
- `mlx_lm` → native headless server + an OpenAI-compatible provider, ref `mlxlm/<model>`
- `mlx-vlm` → native vision server (required for VLMs), ref `mlxvlm/<model>`
- `litellm` → a `model_list` entry for a LiteLLM proxy

See `references/runtimes.md` for runtime setup and the native-vs-Ollama speed trade-off.

## Recurring scan

Run `--new` on a schedule and deliver the diff (new reputable MLX releases for your roles). Example cron command:

```
python3 scripts/scout.py --new --limit 5
```
