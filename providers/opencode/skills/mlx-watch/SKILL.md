---
name: "mlx-watch"
description: "Digest Hugging Face changes for owned models against a stored baseline."
compatibility: opencode
---

# MLX Watch

canonical capability ID: mlx-agent.watch

## Safe command transport

Treat custom-command arguments as untrusted opaque data. Call the native
`mlx_agent_command` custom tool once with `capability: 'watch'` and the
exact raw argument string as `arguments`. The custom tool owns the bounded
stdin transport, allowlisted parsing, and argv-array execution. Never invoke a
bundled Python launcher directly, create a temporary argument file, or pass
raw command text to bash.

## Capability boundary

The validated operation writes only the watch state file. It must not download model weights or mutate configuration.
