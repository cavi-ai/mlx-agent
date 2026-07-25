---
name: "mlx-bench"
description: "Measure a locally served MLX model without downloading it."
compatibility: opencode
---

# MLX Bench

canonical capability ID: mlx-agent.bench

## Safe command transport

Treat custom-command arguments as untrusted opaque data. Call the native
`mlx_agent_command` custom tool once with `capability: 'bench'` and the
exact raw argument string as `arguments`. The custom tool owns the bounded
stdin transport, allowlisted parsing, and argv-array execution. Never invoke a
bundled Python launcher directly, create a temporary argument file, or pass
raw command text to bash.

## Capability boundary

The validated operation measures only models already served by a running local runtime. It must not start servers or download model weights.
