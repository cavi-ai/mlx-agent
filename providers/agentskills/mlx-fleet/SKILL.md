---
name: "mlx-fleet"
description: "Preview and confirm a one-shot per-role router configuration."
---

# MLX Fleet

Resolve `<skill-dir>` as the absolute directory containing this SKILL.md. Never resolve the bundled executable from the shell working directory.

canonical capability ID: mlx-agent.fleet

Render the one-shot per-role router configuration without mutation:

`python3 <skill-dir>/scripts/mlx-agent fleet render --path <router.yaml> --assign <role=repo> --json`

Then request the exact transaction diff and preview hash without confirmation:

`python3 <skill-dir>/scripts/mlx-agent fleet apply --path <router.yaml> --assign <role=repo> --json`

Show that returned diff and preview hash. Do not write configuration files directly. Only after the user explicitly confirms that exact preview, run:

`python3 <skill-dir>/scripts/mlx-agent fleet apply --path <router.yaml> --assign <role=repo> --confirm --preview-hash <preview-hash> --json`

Never download model weights without an explicit confirmation. Report the transaction receipt returned by the CLI.
