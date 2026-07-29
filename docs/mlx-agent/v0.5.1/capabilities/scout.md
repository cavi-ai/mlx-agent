# Scout

Scout discovers MLX models on Hugging Face, bucketed by role for this host: `general`, `coding`, `reasoning`, `vision`, `embedding`, plus `tool-use` membership.

```bash
python3 scripts/mlx-agent discover --role coding --limit 3
python3 scripts/mlx-agent discover --new          # sort by Hub lastModified
python3 scripts/mlx-agent discover --fast         # skip enrichment
python3 scripts/mlx-agent discover --json
```

## What makes the ranking honest

- **Real sizing** — actual quantized byte size from the HF tree API, not a name guess.
- **Context-aware fit** — architecture (layers, KV heads, head dim) from HF config gives each candidate an `estimates.kv` block with the maximum context your RAM supports. `--context N` tightens `fits` to weights + KV at that context.
- **Reasoning detection** — reads the chat template and tags; reasoning models stay out of fast/cheap roles.
- **Quant dedup** — `-4bit / -8bit / -bf16` variants roll up to one logical model and the best quant that fits wins.
- **License and gating** — surfaced before any runtime fetch.
- **Community bench** — when the bundled aggregate has matching numbers for your chip, candidates carry a `community_bench` annotation.

## Filters

`--role`, `--limit`, `--memory-gb`, `--quantization`, `--license`, `--publisher`, `--runtime`, `--exclude-gated`, `--offline`, `--refresh`, and `--context` compose freely. `discover --wire <repo> --target <runtime>` emits a ready config block for a chosen model.
