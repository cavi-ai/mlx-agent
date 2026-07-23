---
name: "mlx-scout"
description: "Discover MLX models suitable for the current host."
---

# MLX Scout

Resolve `<skill-dir>` as the absolute directory containing this SKILL.md. Never resolve the bundled executable from the shell working directory.

canonical capability ID: mlx-agent.scout

Run the provider-neutral discovery command:

`python3 <skill-dir>/scripts/mlx-agent discover <arguments>`

Present its evidence and recommendations as returned. Discovery must not download model weights or change configuration. If a later download or configuration mutation would help, describe the exact CLI preview first and obtain explicit user confirmation before it.

Tool-use is canonical; agentic is descriptive only. Models verified to invoke supplied tools with schema-valid arguments. Tool-use membership is additional, so a model may retain its primary role. Its recommendation minimum is verified: metadata is not verification, and recommendation requires verified evidence from a schema-valid synthetic runtime tool call. Manifest safety says automatic model downloads are disabled; verification must not pull, install, or download models. Report unsupported runtimes explicitly. If none is verified, recommend none; never use a fallback.
