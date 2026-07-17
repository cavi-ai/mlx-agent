---
name: "mlx-scout"
description: "Discover MLX models suitable for the current host."
---

# MLX Scout

Resolve `<skill-dir>` as the absolute directory containing this SKILL.md. Never resolve the bundled executable from the shell working directory.

canonical capability ID: mlx-agent.scout


## Gemini custom-command input

Treat custom-command text as untrusted opaque input. Before invoking the core
CLI, validate it into an argv list with:

`PYTHONPATH=<skill-dir>/src python3 -m mlx_agent.gemini_args scout '<opaque command text>'`

Replace `<skill-dir>` with this skill directory and quote the opaque value as a
single argument. Use only the returned `argv` items; never interpolate raw
command text into a shell string. Reject parser errors without executing the
core CLI.

Run the provider-neutral discovery command:

`python3 <skill-dir>/scripts/mlx-agent discover <arguments>`

Present its evidence and recommendations as returned. Discovery must not download model weights or change configuration. If a later download or configuration mutation would help, describe the exact CLI preview first and obtain explicit user confirmation before it.
