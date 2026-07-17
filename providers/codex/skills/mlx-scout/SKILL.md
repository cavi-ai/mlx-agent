---
name: "mlx-scout"
description: "Discover MLX models suitable for the current host."
---

Use `$mlx-scout` to invoke this installed Codex skill explicitly. Codex does not support custom `/mlx-*` slash commands.

# MLX Scout

Resolve `<skill-dir>` as the absolute directory containing this SKILL.md. Never resolve the bundled executable from the shell working directory.

canonical capability ID: mlx-agent.scout

Run the provider-neutral discovery command:

`python3 <skill-dir>/scripts/mlx-agent discover <arguments>`

Present its evidence and recommendations as returned. Discovery must not download model weights or change configuration. If a later download or configuration mutation would help, describe the exact CLI preview first and obtain explicit user confirmation before it.
