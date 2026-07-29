---
name: mlx-converter
description: "Inventory local GGUF weights on this Mac — what is already converted to MLX, what is still pending, and which copies are redundant — then run confirmation-gated GGUF to MLX conversions."
homepage: https://github.com/cavi-ai/mlx-agent
license: MIT
metadata:
  openclaw:
    emoji: "🔁"
    requires:
      bins: ["python3"]
---

# mlx-converter

Turn GGUF files already on disk into MLX models, and keep the local weight
collection honest. The scan is read-only and stdlib-only; conversion is a
preview-then-confirm job that never downloads, never overwrites, and never
deletes.

## Use when

- Deciding what to convert: which local GGUFs have no MLX output yet.
- Reclaiming disk: finding byte-identical or same-quantization duplicates.
- Converting one GGUF to MLX 4bit or 8bit.
- Checking on a conversion already running.

## Run

- Inventory the well-known roots: `python3 scripts/mlx_converter.py scan`
- Specific directories: `python3 scripts/mlx_converter.py scan --gguf-root ~/models --mlx-root ~/mlx`
- Only what still needs converting: `python3 scripts/mlx_converter.py scan --pending-only`
- Machine-readable: `python3 scripts/mlx_converter.py scan --json`
- Preview a conversion: `python3 scripts/mlx_converter.py start --gguf ~/models/model-Q4_K_M.gguf --q-bits 4`
- Run the reviewed plan: `... start --gguf PATH --q-bits 4 --confirm --preview-hash HASH`
- Poll jobs: `python3 scripts/mlx_converter.py status`

`scan` needs nothing but Python. Conversion needs `mlx-lm` on PATH plus
`torch`, `transformers`, and `gguf` importable by the same interpreter; the
skill reports what is missing and never installs it for you.

## Reading the scan

Each GGUF gets a status:

- `pending` — no MLX output found. This is the convert list.
- `converted` — an MLX output exists. `evidence` says how it was matched:
  `provenance` (the marker this tool wrote — exact), `receipt` (a convert
  receipt), or `name` (a name match — weaker, confirm before deleting anything).
- `shard` — a non-first shard of a split model; convert the `-00001-of-` file.
- `companion` — a projector (`mmproj`) or similar sidecar. Not convertible on
  its own; it belongs to a base model.

Duplicate groups come in two kinds:

- `exact` — same bytes, or same structure at the same quantization. Everything
  outside `keep` is redundant and its size counts toward `reclaimable_bytes`.
- `variant` — the same model at different quantization levels. Reported so the
  choice is visible, never recommended for removal.

Duplicate identity uses a size-plus-head-plus-tail signature rather than a full
digest, because full digests of multi-gigabyte weights are not worth the I/O.
Pass `--no-signature` to skip it entirely and fall back to header structure.

## Converting

Conversion is two steps in one job: `transformers` dequantizes the GGUF back to
Hugging Face weights, then `mlx_lm.convert` quantizes those to MLX. The
intermediate checkpoint is temporary and removed afterwards, but it is
full-precision — budget roughly 2x the original model's fp16 size in free disk
while the job runs.

Quality is capped by the GGUF's own quantization: a Q4 GGUF converted to MLX
4bit has been quantized twice. When the original fp16 weights are available,
convert those instead (`start --repo publisher/model`).

The job writes `mlx-converter.json` into the output directory recording which
GGUF produced it, so later scans can pair them exactly.

## Removing redundant copies

The scan only reports. Deleting is the user's call, and the file paths in
`redundant` are the exact candidates. Move rather than delete when unsure.
