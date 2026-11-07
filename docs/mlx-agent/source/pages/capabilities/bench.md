# Bench

Bench measures a model already served by a running local runtime. One warm-up plus N timed streaming runs report first-content latency and available token rates. The timeout covers the warm-up and all timed runs; warm-up results are excluded from statistics.

```bash
python3 scripts/mlx-agent bench run --repo <repo> --runtime mlx_lm
python3 scripts/mlx-agent bench run --repo <repo> --runtime ollama --runs 5 --json
python3 scripts/mlx-agent bench aggregate --exports ./bench-exports --out ./aggregate.json
```

## Boundaries

- Measures only what is already serving: `model_not_serving` otherwise. Never starts servers (that is `serve`'s job), never downloads.
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

`bench run --export <file.jsonl>` appends a bounded, anonymized schema 2.0 record: repo, chip, runtime, measurement method, requested output-token limit, measured prompt count, and available rates/timings. It contains no hostnames, paths, or prompts. A missing chip prevents export.

`bench aggregate` emits schema 2.0 and retains the newest record per (repo, chip, runtime, measurement method, requested output limit, measured prompt count). Each entry's `samples` counts retained export records, not timed inference runs. Different servers, methods, and workload bounds remain separate. Schema 1.0 exports remain readable and appear under `legacy_unclassified`; historical token rates are not reclassified as server timings.

Community lookup omits a repo/chip annotation when multiple groups match, rather than selecting an arbitrary runtime or averaging unlike results. Existing schema 1.0 community files remain readable and their methodology is labeled legacy.

Anonymous exports do not bind weight hashes, runtime revisions, sampler settings, or cache state. For a controlled comparison, record those identities separately and use the same workload and hardware conditions. Repeat prompts may reuse runtime caches; these measurements do not establish cold-start performance.
