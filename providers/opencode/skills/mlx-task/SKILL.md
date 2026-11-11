---
name: "mlx-task"
description: "Run, track and collect durable local inference tasks."
compatibility: opencode
---

# MLX Task

canonical capability ID: mlx-agent.task

## Safe command transport

Treat custom-command arguments as untrusted opaque data. Call the native
`mlx_agent_command` custom tool once with `capability: 'task'` and the
exact raw argument string as `arguments`. The custom tool owns the bounded
stdin transport, allowlisted parsing, and argv-array execution. Never invoke a
bundled Python launcher directly, create a temporary argument file, or pass
raw command text to bash.

## Capability boundary

List kinds with `capabilities --json`, start with an absolute `task-request` file, then use status, result, list, cancel or verify-model by task ID. Tasks use only existing local models and must not download model weights or install a runtime.
