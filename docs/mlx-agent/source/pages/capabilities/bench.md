# Bench

Bench measures a model already served by a running local runtime. One warm-up plus N timed streaming runs report first-content latency and available token rates. The timeout covers the warm-up and all timed runs; warm-up results are excluded from statistics.

```bash
python3 scripts/mlx-agent bench run --repo <repo> --runtime mlx_lm
python3 scripts/mlx-agent bench run --repo <repo> --runtime ollama --runs 5 --json
python3 scripts/mlx-agent bench aggregate --exports ./bench-exports --out ./aggregate.json
```

## Workloads

`--workload` selects what each run sends; `chat`, the default, is the measurement above.

| Workload | Each run sends | Reports |
|---|---|---|
| `prefix` | About 16k tokens of shared context, then a different question | `first_ttft_ms` for the warm-up, timed `ttft_ms`, and server-reported `cached_tokens` |
| `prefill` | About 32k tokens that share no prefix with other runs | `ttft_ms`, `prefill_toks`, `prompt_tokens` |
| `concurrent` | `--concurrency` streams at once (2–8, default 4) | Available per-stream rates; `aggregate_end_to_end_toks`, every stream's reported tokens over the round's wall time |
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
Concurrent throughput is unavailable if any stream lacks completion counts.
The older `aggregate_decode_toks` field remains `null`: round wall time does
not establish aggregate server decode timing.

## Boundaries

- Measures only what is already serving: `model_not_serving` otherwise. Never starts servers (that is `serve`'s job), never downloads. A JIT endpoint loads its model for a bench request as it does for any request; bench never unloads it.
- Loopback-only endpoints, absolute deadlines, bounded responses — the same envelope as verification.
- Decode and prefill tok/s require matching server token counters and stage durations, such as Ollama's usage fields. Reported cached prompt tokens are excluded from uncached prefill throughput. Without these timings, the stage rate is `null` (`n/a` in text output).
- End-to-end tok/s uses the server's reported completion-token count divided by complete client elapsed time, including prompt processing and stream completion. It is reported separately from decode speed.
- TTFT measures time to the first nonempty content event. Queueing, transport and cache behavior can affect it; it does not establish prefill speed. Usage-only responses have no measured TTFT.
- Stream chunks are never counted as tokens. Missing prompt counts remain `null`; no fixed prompt-token estimate is substituted.
- `measurement_method` identifies `server_timings`, `client_wall_clock`, `latency_only`, or `mixed` samples. A metric missing from any timed run has no aggregate value. Decode spread is unavailable without decode timing in every run.
- Per-run samples retain client elapsed time and valid server usage. Peak memory is currently unmeasured and remains `null`.

## Evidence

Results emit `runtime_measured` evidence (`bench-v2`). This can mean latency alone was measured; inspect the method and nullable rates before interpreting speed. `adopt start --measure` runs bench on verified shortlist candidates automatically. Measurement is not a model-quality verdict or a comparative performance result.

## Export and aggregate

`bench run --export <file.jsonl>` appends a bounded schema 2.0 record: repo, chip, runtime, measurement method, requested output-token limit, measured prompt count, and available rates/timings. Export fields omit prompts, logs, and hostnames. The supplied model identifier is retained verbatim and can identify a private repository or local path. A missing chip prevents export.

`bench aggregate` emits schema 2.0 and retains the newest record per (repo, chip, runtime, measurement method, requested output limit, measured prompt count). Each entry's `samples` counts retained export records, not timed inference runs. Different servers, methods, and workload bounds remain separate. Schema 1.0 exports remain readable and appear under `legacy_unclassified`; historical token rates are not reclassified as server timings.

Community lookup omits a repo/chip annotation when multiple groups match, rather than selecting an arbitrary runtime or averaging unlike results. Existing schema 1.0 community files remain readable and their methodology is labeled legacy.

Anonymous exports do not bind weight hashes, runtime revisions, sampler settings, or cache state. For a controlled comparison, record those identities separately and use the same workload and hardware conditions. Repeat prompts may reuse runtime caches; these measurements do not establish cold-start performance.
