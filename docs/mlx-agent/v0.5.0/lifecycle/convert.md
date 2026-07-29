# Convert

Convert quantizes a model to MLX (4bit or 8bit) as a receipt-tracked batch job, from a cached Hugging Face repo or from a GGUF file already on disk.

```bash
python3 scripts/mlx-agent convert start --repo <repo> --q-bits 4
python3 scripts/mlx-agent convert start --repo <repo> --q-bits 4 --confirm --preview-hash <hash>
python3 scripts/mlx-agent convert status
```

## Gates

- Source model in the Hugging Face cache (never downloads).
- `mlx_lm.convert` already installed (never installs).
- Fresh output path (never overwrites).
- One job at a time per receipts root — conversions are GPU-bound.

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

A directory counts as an MLX output only when its `config.json` carries a `quantization` block or this tool left an `mlx-converter.json` marker, so ordinary PyTorch checkpoints in the cache are never mistaken for conversions.

Duplicates come in two strengths. `exact` groups are the same bytes, or the same header structure at the same quantization: everything outside `keep` is redundant and counts toward `reclaimable_bytes`. `variant` groups are the same model at different quantization levels — reported for visibility, never recommended for removal. Identity uses size plus the first and last mebibyte rather than a full digest, because hashing eighty gigabytes to answer a dedupe question is not worth the I/O. `scan` only reports; deleting stays with you.

Conversion runs two steps in one job: `transformers` dequantizes the GGUF back to Hugging Face weights, then `mlx_lm.convert` quantizes those to MLX. The intermediate checkpoint is full precision and removed afterwards, so budget roughly twice the source model's fp16 size in free disk while the job runs. It needs `torch`, `transformers`, and `gguf` importable by the same interpreter; missing pieces are named and never installed for you.

Quality is capped by the source: a Q4 GGUF converted to MLX 4bit has been quantized twice. Convert the original fp16 weights when you have them.

The output carries `mlx-converter.json` recording which GGUF produced it, so later scans pair the two exactly.
