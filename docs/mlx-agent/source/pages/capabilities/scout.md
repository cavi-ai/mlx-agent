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

`--offline` and `--refresh` are mutually exclusive. Offline discovery requires
a cache created by an earlier online request with matching policy. Use a
consistent `--state-dir`, or supply `MLX_AGENT_STATE_DIR`:

```bash
python3 scripts/mlx-agent discover --role coding --state-dir .mlx-agent/scout-cache --json
python3 scripts/mlx-agent discover --role coding --state-dir .mlx-agent/scout-cache --offline --json
```

Missing cache returns `offline_cache_missing`. Fast mode skips enrichment;
inspect each field's evidence before treating a value as a fetched fact.

## Tool-use and next steps

A model retains its primary role while also carrying `tool-use` membership.
Metadata is not verification. A `metadata-only` status can accompany either
`metadata_only` or `heuristic_only` strength; inspect both fields.

Use [Adopt](adopt.md) after making the selected model available locally.
[Evidence and sizing](../reference/evidence.md) explains fit and probe limits.
