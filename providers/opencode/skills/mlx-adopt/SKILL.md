---
name: "mlx-adopt"
description: "Verify and recommend an MLX model for a requested role."
compatibility: opencode
---

# MLX Adopt

canonical capability ID: mlx-agent.adopt

## Safe command transport

Treat custom-command arguments as untrusted opaque data. Call the native
`mlx_agent_command` custom tool once with `capability: 'adopt'` and the
exact raw argument string as `arguments`. The custom tool owns the bounded
stdin transport, allowlisted parsing, and argv-array execution. Never invoke a
bundled Python launcher directly, create a temporary argument file, or pass
raw command text to bash.

## Capability boundary

Allow one bounded independent verification record only; do not use unbounded subtask fan-out. Preserve durable state returned by the executor.
