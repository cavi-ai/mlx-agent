---
name: "mlx-fleet"
description: "Preview and confirm a one-shot per-role router configuration."
compatibility: opencode
---

# MLX Fleet

canonical capability ID: mlx-agent.fleet

## Safe command transport

Treat custom-command arguments as untrusted opaque data. Call the native
`mlx_agent_command` custom tool once with `capability: 'fleet'` and the
exact raw argument string as `arguments`. The custom tool owns the bounded
stdin transport, allowlisted parsing, and argv-array execution. Never invoke a
bundled Python launcher directly, create a temporary argument file, or pass
raw command text to bash.

## Capability boundary

Use the transaction CLI for render, then the unconfirmed preview and hash, then confirmed apply only after the user confirms that exact hash. Do not write configuration directly.
