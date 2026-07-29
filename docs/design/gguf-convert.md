# Design: GGUF sources for `mlx-agent convert`

Status: implemented. Version target: 0.5.0. Supersedes the "GGUF sources" v1 exclusion in `convert.md`.

## Problem

GGUF is what most people already have on disk — pulled by LM Studio, llama.cpp, or a one-off download. `mlx_lm.convert` only reads Hugging Face format, so the common local task ("I have this GGUF, give me MLX") has no path through the pack. Users also lose track of which of their GGUFs they already converted, and accumulate duplicate quantizations that cost tens of gigabytes.

## Decision: read-only inventory plus a two-step conversion job

Two separable pieces, both CLI-only like the rest of the lifecycle commands:

1. `convert scan` — a read-only inventory. Bounded GGUF header parse (stdlib `struct`, no `gguf` package needed), pairing against MLX outputs, and duplicate grouping. Reports; never mutates.
2. `convert start --gguf PATH` — one spawned job that dequantizes via `transformers` and quantizes via `mlx_lm.convert`, reusing the existing preview → confirm → receipt machinery unchanged.

## Inventory

`scan` walks `--gguf-root` directories (default: well-known local model directories that exist), reads each file's bounded header, and classifies:

- `pending` — no MLX output found; the convert list.
- `converted` — matched by `provenance` (the `mlx-converter.json` marker this tool writes), `receipt`, or `name`. Evidence strength is reported, because only the first two are exact.
- `shard` — a non-first `-000NN-of-000NN` file; the loader reads the rest from shard one.
- `companion` — `mmproj`/projector sidecars, which carry the base model's `general.name` and are not standalone conversions.

A directory counts as an MLX output only when its `config.json` has a `quantization` block or it holds our provenance marker. Plain PyTorch checkpoints in the HF cache also have `config.json` plus safetensors and must not be read as conversions.

## Duplicates

Two groups, deliberately different in strength:

- `exact` — same content signature, or same header structure at the same quantization. Everything outside `keep` is redundant; `reclaimable_bytes` is the sum. Keeper preference: an already-converted copy, then the oldest path.
- `variant` — same model at different quantizations. Reported for visibility, never recommended for removal: choosing between Q4 and Q8 is the user's call.

Identity uses size + first MiB + last MiB rather than a full digest. Full digests of 80GB weights cost minutes of I/O for a dedupe answer; the collision risk for weight files is not worth that.

## Conversion

`gguf_runner.py` is spawned by absolute path with the current interpreter and imports nothing from `mlx_agent`, so it works whether the package is installed or merely on `sys.path`. Steps: `transformers.AutoModelForCausalLM.from_pretrained(dir, gguf_file=name)` → `save_pretrained` → `mlx_lm.convert -q --q-bits N`. The fp16 intermediate is removed unless `--keep-intermediate`.

## Hard gates

- Path ends in `.gguf`, exists, parses, and is not a non-first shard.
- `mlx_lm.convert` on PATH, plus `torch`, `transformers`, `gguf` importable (`runtime_not_installed`; never installs).
- Output path must not exist; one job at a time per receipts root.
- `--confirm --preview-hash` as with every other convert plan.

## Receipts

Convert receipts gain `slug` (filename-safe, replaces the `repo.split("/")` filename derivation that assumed an HF repo) and `source` (`{kind: "hf-cache" | "gguf", ...}`). Both are optional in the schema so pre-existing receipts still read.

## Files touched

`src/mlx_agent/gguf.py` (new), `src/mlx_agent/gguf_runner.py` (new), `convert.py`, `cli.py`, `schemas/convert-receipt.schema.json`, `skills/mlx-converter/` (new), `tests/unit/test_gguf.py`, README/CHANGELOG.

## Exclusions

- Deletion of duplicates. The scan reports paths; removal stays with the user.
- Vision projectors (`mmproj`) as conversion inputs.
- Quant recipes beyond `--q-bits` (group size, dtype).
