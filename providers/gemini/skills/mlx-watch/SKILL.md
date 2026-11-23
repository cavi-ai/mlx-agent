---
name: "mlx-watch"
description: "Digest Hugging Face changes for owned models against a stored baseline."
---

# MLX Watch

Resolve `<plugin-root>` as the absolute directory two levels above the directory containing this SKILL.md; it holds `scripts/` and `src/`. Never resolve the bundled executable from the shell working directory.

canonical capability ID: mlx-agent.watch

Record or compare the owned-model baseline:

`python3 <plugin-root>/scripts/mlx-agent watch snapshot --json`
`python3 <plugin-root>/scripts/mlx-agent watch diff --json`

Present the classified findings as returned. Watch writes only its own state file and never downloads model weights or changes configuration.
