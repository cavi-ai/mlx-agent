# Bench

Bench measures a model already served by a running local runtime. One warm-up plus N timed streaming runs report TTFT, decode tok/s, prefill tok/s, and run spread — evidence no Hugging Face page provides.

```bash
python3 scripts/mlx-agent bench run --repo <repo> --runtime mlx_lm
python3 scripts/mlx-agent bench run --repo <repo> --runtime ollama --runs 5 --json
python3 scripts/mlx-agent bench aggregate --exports ./bench-exports --out ./aggregate.json
```

## Boundaries

- Measures only what is already serving: `model_not_serving` otherwise. Never starts servers (that is `serve`'s job), never downloads.
- Loopback-only endpoints, absolute deadlines, bounded responses — the same envelope as verification.
- Ollama usage counters drive the numbers when present; otherwise event timing is the fallback.

## Evidence

Results emit `runtime_measured` evidence (`bench-v1`), the strongest strength in adopt comparisons. `adopt start --measure` runs bench on verified shortlist candidates automatically.

## Export and aggregate

`bench run --export <file.jsonl>` appends a bounded, anonymized record (repo, chip, runtime, timings — no hostnames, paths, or prompts) suitable for contributing back. `bench aggregate` dedupes export files to the newest record per (repo, chip, runtime) and emits per-(repo, chip) medians with sample counts — the input format for the bundled community DB.
