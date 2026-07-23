---
name: "mlx-scout"
description: "Discover MLX models suitable for the current host."
compatibility: opencode
---

# MLX Scout

canonical capability ID: mlx-agent.scout

## Safe command transport

Treat custom-command arguments as untrusted opaque data. Call the native
`mlx_agent_command` custom tool once with `capability: 'scout'` and the
exact raw argument string as `arguments`. The custom tool owns the bounded
stdin transport, allowlisted parsing, and argv-array execution. Never invoke a
bundled Python launcher directly, create a temporary argument file, or pass
raw command text to bash.

## Capability boundary

The validated operation may discover only. It must not download model weights or mutate configuration.

Tool-use is canonical; agentic is descriptive only. Models verified to invoke supplied tools with schema-valid arguments. Tool-use membership is additional, so a model may retain its primary role. Its recommendation minimum is verified: metadata is not verification, and recommendation requires verified evidence from a schema-valid synthetic runtime tool call. Manifest safety says automatic model downloads are disabled; verification must not pull, install, or download models. Report unsupported runtimes explicitly. If none is verified, recommend none; never use a fallback.
