# LoRA and fuse

LoRA trains an adapter on a cached base model; fuse merges the adapter back into standalone weights.

## Train

```bash
python3 scripts/mlx-agent lora start --repo <repo> --data ./my-dataset --iters 1000
python3 scripts/mlx-agent lora start --repo <repo> --data ./my-dataset --confirm --preview-hash <hash>
python3 scripts/mlx-agent lora status
```

The dataset is validated before the preview even renders: `<dir>/train.jsonl` (optional `valid.jsonl`), each line a JSON object with a `text` string or a `messages` array of `{role, content}`. Bounded hyperparameters: `--iters` (1–100000), `--batch-size` (1–64), `--learning-rate` (1e-6–1e-2), `--num-layers` (-1–128).

## Fuse

```bash
python3 scripts/mlx-agent fuse start --repo <repo> --adapter ./adapter
python3 scripts/mlx-agent fuse start --repo <repo> --adapter ./adapter --confirm --preview-hash <hash>
python3 scripts/mlx-agent fuse status
```

Fuse validates the adapter (`adapter_config.json` present and parseable) and renders the exact `mlx_lm.fuse` argv under the same gates: cached base, installed runtime, fresh output path, one job at a time.

## Serve the result

```bash
python3 scripts/mlx-agent serve start --repo <repo> --runtime mlx_lm --adapter-path ./adapter
```

Train on a quantized base, fuse to keep the quant, serve either the adapter or the fused model.
