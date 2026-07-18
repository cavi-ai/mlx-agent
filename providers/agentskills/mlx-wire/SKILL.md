---
name: "mlx-wire"
description: "Preview and confirm wiring a chosen MLX model."
---

# MLX Wire

Resolve `<skill-dir>` as the absolute directory containing this SKILL.md. Never resolve the bundled executable from the shell working directory.

canonical capability ID: mlx-agent.wire

Use the structured CLI to inspect the target configuration without mutation:

`python3 <skill-dir>/scripts/mlx-agent wire render <model> --target <target> --path <config-path> --json`

Then request the exact transaction diff and preview hash without confirmation. This command is intentionally non-mutating and exits nonzero while it waits for confirmation:

`python3 <skill-dir>/scripts/mlx-agent wire apply <model> --target <target> --path <config-path> --json`

Show that returned diff and preview hash. Do not write configuration files directly. Only after the user explicitly confirms that exact preview, run:

`python3 <skill-dir>/scripts/mlx-agent wire apply <model> --target <target> --path <config-path> --confirm --preview-hash <preview-hash> --json`

Never download model weights without an explicit confirmation. Report the transaction receipt returned by the CLI.
