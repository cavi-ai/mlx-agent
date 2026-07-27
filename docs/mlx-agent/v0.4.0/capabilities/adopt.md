# Adopt

Adopt is the resumable discover → verify → recommend workflow with durable evidence. State lives in one JSON handoff file; every phase persists atomically and resumes where it stopped.

```bash
python3 scripts/mlx-agent adopt start --state ./adopt.json --role coding
python3 scripts/mlx-agent adopt resume --state ./adopt.json
python3 scripts/mlx-agent adopt status --state ./adopt.json
```

## Verification evidence

Each shortlisted candidate gets an evidence record with an ordered strength:

| Strength | Meaning |
| --- | --- |
| `runtime_measured` | Benched on this host (TTFT, tok/s). |
| `runtime_tested` | Generated and probed against a local runtime. |
| `runtime_inventory` | Present in a runtime inventory, not generated. |
| `metadata_only` | Hub metadata inspected; model not local. |
| `heuristic_only` | Name/metadata heuristics only. |

## Role probes

Deterministic, synthetic, never LLM-judged: `coding-v1` (AST parse + sandboxed exec), `reasoning-v1` (exact numeric answer), `vision-v1` (OCR of a bundled synthetic image), `embedding-v1` (cosine ordering), and `tool-use-v1` (schema-valid synthetic tool call). A failed probe rejects the candidate for that role; an unsupported runtime is recorded, never penalized.

`adopt start --measure` adds a bounded measure phase between verify and compare that benches verified candidates and upgrades their evidence to `runtime_measured`.

Tool-use is canonical: metadata is never verification. Only a schema-valid synthetic runtime tool call recommends a model as tool-use capable.
