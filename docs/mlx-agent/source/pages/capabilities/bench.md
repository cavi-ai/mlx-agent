# Bench

Bench measures a model already served by a running local runtime. One warm-up plus N timed streaming runs report TTFT, decode tok/s, prefill tok/s, and run spread — evidence no Hugging Face page provides.

```bash
python3 scripts/mlx-agent bench run --repo <repo> --runtime mlx_lm
python3 scripts/mlx-agent bench run --repo <repo> --runtime ollama --runs 5 --json
python3 scripts/mlx-agent bench aggregate --exports ./bench-exports --out ./aggregate.json
```

## Workloads

`--workload` selects what each run sends; `chat`, the default, is the measurement above.

| Workload | Each run sends | Reports |
|---|---|---|
| `prefix` | About 16k tokens of shared context, then a different question | `first_ttft_ms` (the warm-up, which reads the prefix uncached), `ttft_ms` with the prefix cached, `cached_tokens` |
| `prefill` | About 32k tokens that share no prefix with other runs | `ttft_ms`, `prefill_toks`, `prompt_tokens` |
| `concurrent` | `--concurrency` streams at once (2–8, default 4) | Per-stream `decode_toks`; `aggregate_decode_toks`, every stream's tokens over the run's wall time |
| `cold` | One request, without warm-up, to an owned JIT endpoint whose model is unloaded | `ttft_ms` including the model load |

```bash
python3 scripts/mlx-agent bench run --repo <repo> --runtime mlx_lm --base-url http://127.0.0.1:8090 --workload prefix --json
python3 scripts/mlx-agent serve unload --port 8090 --json
python3 scripts/mlx-agent bench run --repo <repo> --runtime mlx_lm --base-url http://127.0.0.1:8090 --workload cold --runs 1 --json
```

`--base-url` measures a runtime at another loopback origin than its standard
port. Each long prompt begins with a tag unique to the invocation, so a later
invocation never reuses an earlier one's prompt cache. Token counts are
approximate; `prompt_tokens` and `cached_tokens` are the server's counts when it
reports them (`usage.prompt_tokens_details.cached_tokens`). `cold` needs
`--runs 1` and the endpoint's `--receipts-dir` when it is not the current
directory, and refuses a loaded model with `model_loaded`. To measure gateway
overhead, run a workload against a JIT endpoint and against the `worker_port`
that `serve status` reports while its model is loaded, with a model id that the
worker's `/v1/models` lists. Only `chat` emits evidence and accepts `--export`.

## Boundaries

- Measures only what is already serving: `model_not_serving` otherwise. Never starts servers (that is `serve`'s job), never downloads. A JIT endpoint loads its model for a bench request as it does for any request; bench never unloads it.
- Loopback-only endpoints, absolute deadlines, bounded responses — the same envelope as verification.
- Ollama usage counters drive the numbers when present; otherwise event timing is the fallback.

## Evidence

Results emit `runtime_measured` evidence (`bench-v1`), the strongest strength in adopt comparisons. `adopt start --measure` runs bench on verified shortlist candidates automatically.

## Export and aggregate

`bench run --export <file.jsonl>` appends a bounded, anonymized record (repo, chip, runtime, timings — no hostnames, paths, or prompts) suitable for contributing back. `bench aggregate` dedupes export files to the newest record per (repo, chip, runtime) and emits per-(repo, chip) medians with sample counts — the input format for the bundled community DB.
