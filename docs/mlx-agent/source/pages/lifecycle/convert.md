# Convert

Convert quantizes a model to MLX (4bit or 8bit) as a receipt-tracked batch job, from a cached Hugging Face repo or from a GGUF file already on disk.

```bash
python3 scripts/mlx-agent convert start --repo <repo> --q-bits 4
python3 scripts/mlx-agent convert start --repo <repo> --q-bits 4 --confirm --preview-hash <hash>
python3 scripts/mlx-agent convert status
```

## Gates

- Source checkpoint already local: cached repository, explicit source path, or GGUF file. Conversion does not download it.
- Selected converter backend already installed (the default is `mlx_lm.convert`).
- Fresh output path (never overwrites).
- Conversion receipt checks and a shared compute lease prevent conflicting heavy jobs.

The preview renders the exact `mlx_lm.convert` argv before anything runs. `convert status` cross-checks the receipt against the live process and records the exit once (`done` when the output directory exists, `failed` otherwise).

Result lands at `<model>-MLX-<bits>bit` by default, ready to serve or wire.

## GGUF sources

Most local weights arrive as GGUF. `convert scan` inventories them; `convert start --gguf` converts one.

```bash
python3 scripts/mlx-agent convert scan
python3 scripts/mlx-agent convert scan --gguf-root ~/models --pending-only --json
python3 scripts/mlx-agent convert start --gguf ~/models/model-Q4_K_M.gguf --q-bits 4
```

`scan` is read-only and needs nothing but Python. It parses bounded GGUF headers under each `--gguf-root` (repeatable; well-known local model directories by default) and gives every file a status:

| Status | Meaning |
| --- | --- |
| `pending` | No MLX output found — the convert list. |
| `converted` | An MLX output exists; `evidence` says whether it matched by `provenance`, `receipt`, or `name`. |
| `shard` | A non-first shard of a split model; convert the `-00001-of-` file. |
| `companion` | A projector (`mmproj`) or similar sidecar, not convertible alone. |

A directory counts as an MLX output only when its `config.json` carries a `quantization` block or this tool left an `mlx-converter.json` marker, so ordinary PyTorch checkpoints in the cache are never mistaken for conversions. Cached snapshots of `mlx-community` repositories also count, including unquantized (bf16) ones.

Each output carries a `task` label and, except Strata profiles, a `backend` object: the backend that runs it for that task (`mlx-lm`, `mlx-vlm`, `mlx-audio`, …) and whether it is `installed`. A config without a `model_type` is matched by the folder's (or cached repository's) leading word, as the backends themselves do; Kokoro resolves to `mlx-audio` text to speech.

Duplicate groups use two labels. `exact` groups match a sampled content signature
or the same header structure and quantization; members outside `keep` count
toward `reclaimable_bytes`. This classification is not a full-file byte comparison.
The content signature uses size plus the first and last mebibyte, so different
interior bytes can share it. `variant` groups have different quantizations and
are informational, with no removal recommendation. Review identity and any
conversion provenance before deleting a copy. Scan only reports.

MLX model folders are grouped too (`format: "mlx"`). An `exact` MLX group holds
the same weight files at the same sizes and content signatures; without
signatures it also needs the same model name and bit width. `keep` is a
converter output first, then a Hugging Face cache snapshot (named by its
repository), then the oldest copy; cache revisions sharing blobs count as one
copy. `variant` MLX groups share a model name (without its parameter count)
and architecture at different bit widths. Every group carries a `group_id`.

Ollama stores at or below the scan roots (a folder holding `blobs` beside
`manifests` or `manifests-v2`) are read too and listed under `ollama`: name,
manifest `path`, `format` (`gguf` or `safetensors`), quantization and bytes. A
GGUF model joins the GGUF groups by its model blob; when it duplicates a file,
the Ollama copy is kept, since only Ollama removes its blobs. A safetensors
model joins the MLX `variant` groups through its `config.json` layer. Models
listed in both manifest trees count once, and models sharing a blob are one copy.

Conversion runs two steps in one job: `transformers` dequantizes the GGUF back to Hugging Face weights, then `mlx_lm.convert` quantizes those to MLX. The intermediate checkpoint is full precision and removed afterwards, so budget roughly twice the source model's fp16 size in free disk while the job runs. It needs `torch`, `transformers`, and `gguf` importable by the same interpreter; missing pieces are named and never installed for you.

Qwen3.5-family GGUFs (`qwen35`, including Qwen3.6 and Qwen3.8 checkpoints) convert through a bundled port instead, which needs only `gguf` and `mlx` and writes no intermediate checkpoint. The port writes the MLX-VLM `qwen3_5_text` layout, drops the multi-token-prediction block and rebuilds the tokenizer from the GGUF vocabulary. When the GGUF's folder holds exactly one Qwen3-VL projector (`mmproj`) that matches the model's width, the plan names it and the output becomes the `qwen3_5` vision-language model. These outputs are served through the `mlx-vlm` backend. A GGUF holding only a multi-token-prediction block is a speculative-decoding drafter and is refused at plan time.

Quality is capped by the source: a Q4 GGUF converted to MLX 4bit has been quantized twice. Convert the original fp16 weights when you have them.

The output carries `mlx-converter.json` recording its GGUF source, so later
scans can associate the output with that source using provenance evidence.

## Other checkpoints and backends

Use [Intake](intake.md) to identify the model type and required backend.
`--backend` and `--model-type` select a declared converter; install an optional
backend through a separately reviewed [backend plan](backends.md). Use
`--subfolder` for a cached checkpoint folder, or `--source-path` for the exact
downloaded directory associated with `--repo`. Conversion does not fetch missing
weights. GGUF conversion uses `--gguf` rather than an optional backend override.

After conversion finishes, use [Serve](serve.md) for a compatible HTTP model or
[Local inference](local-inference.md) for supported speech, classification,
image, and video tasks. Inspect status and logs before relying on outputs.
