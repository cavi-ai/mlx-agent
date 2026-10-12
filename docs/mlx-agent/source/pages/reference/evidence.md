# Evidence and sizing

Discovery helps choose what to try. Verification records what a local runtime
actually did. A recommendation retains its evidence strength so a caller can
distinguish a metadata candidate from a tested or measured model.

## Discovery signals

- **Weight size:** enriched discovery reads quantized file bytes from the Hugging
  Face tree API. Missing enrichment and fast mode can leave estimates or heuristics.
- **Context fit:** architecture fields such as layers, KV heads, and head dimension
  feed the KV-cache estimate. `--context N` checks weights plus estimated KV memory
  at that context. This is a capacity estimate, not a reservation or an OOM guarantee.
- **Reasoning:** chat-template and tag signals take precedence over name heuristics.
  Inspect their provenance when assigning a model to a latency-sensitive role.
- **Quantization:** related quant variants are grouped and a fitting variant is
  selected. Different quantizations can have different quality.
- **License and access:** license and gating metadata are surfaced for review;
  discovery does not accept Hub terms or establish permitted use.
- **Community measurements:** matching chip/model aggregates annotate candidates
  as `community_bench`. They do not change discovery ranking and do not measure
  your current runtime configuration.

## Evidence strengths

| Strength | What happened |
| --- | --- |
| `runtime_measured` | A local benchmark recorded timing and throughput. |
| `runtime_tested` | A local generation or deterministic role probe ran. |
| `runtime_inventory` | A runtime listed the model; no generation was tested. |
| `metadata_only` | Hub metadata was inspected. |
| `heuristic_only` | Classification used heuristics without stronger evidence. |

Verification status and evidence strength are separate fields. For example,
`metadata-only` can accompany either `metadata_only` or `heuristic_only` strength.
An unsupported probe is recorded as `unsupported-runtime`; it does not establish
either a model failure or a successful role check.

## Role probes and limits

The probes use fixed synthetic inputs: coding checks Python AST and bounded
execution, reasoning checks an answer, vision checks OCR, embedding checks cosine
ordering, and tool-use checks a schema-valid synthetic runtime tool call.
Tool-use metadata is not verification. Models retain a primary role while also
carrying tool-use membership.

Passing these bounded checks does not establish production quality, a security
boundary, or correctness on your workload. Measure with [Bench](../capabilities/bench.md)
and evaluate representative tasks before relying on a model. Changing weights,
quantization, templates, adapters, or runtime settings can change behavior.
