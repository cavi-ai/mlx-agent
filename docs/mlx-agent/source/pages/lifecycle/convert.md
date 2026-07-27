# Convert

Convert quantizes a cached model to MLX (4bit or 8bit) as a receipt-tracked batch job.

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
