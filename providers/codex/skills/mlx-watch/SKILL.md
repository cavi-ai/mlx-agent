---
name: "mlx-watch"
description: "Digest Hugging Face changes for owned models against a stored baseline."
---

Use `$mlx-agent:mlx-watch` to invoke this installed Codex skill explicitly. Codex does not support custom `/mlx-*` slash commands.

# MLX Watch

Resolve `<skill-dir>` as the absolute directory containing this SKILL.md. Never resolve the bundled executable from the shell working directory.

canonical capability ID: mlx-agent.watch

Record or compare the owned-model baseline:

`python3 <skill-dir>/scripts/mlx-agent watch snapshot --json`
`python3 <skill-dir>/scripts/mlx-agent watch diff --json`

Present the classified findings as returned. Watch writes only its own state file and never downloads model weights or changes configuration.
