---
name: "mlx-doctor"
description: "Diagnose local model inventories, wiring drift, and endpoint health."
compatibility: opencode
---

# MLX Doctor

canonical capability ID: mlx-agent.doctor

## Safe command transport

Treat custom-command arguments as untrusted opaque data. Call the native
`mlx_agent_command` custom tool once with `capability: 'doctor'` and the
exact raw argument string as `arguments`. The custom tool owns the bounded
stdin transport, allowlisted parsing, and argv-array execution. Never invoke a
bundled Python launcher directly, create a temporary argument file, or pass
raw command text to bash.

## Capability boundary

The validated operation is read-only diagnostics. Prune is irreversible and requires the user's explicitly reviewed preview hash.
