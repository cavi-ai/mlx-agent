---
name: mlx-wire
description: Preview and confirm wiring a chosen MLX model.
---

# MLX Wire

canonical capability ID: mlx-agent.wire

Use the structured CLI to render a non-mutating configuration preview:

`python3 ../../../scripts/mlx-agent wire render <model> --target <target> --path <config-path> --json`

Show the returned preview and its hash. Do not write configuration files directly. Only after the user explicitly confirms that exact preview, run:

`python3 ../../../scripts/mlx-agent wire apply <model> --target <target> --path <config-path> --confirm --preview-hash <preview-hash> --json`

Never download model weights without an explicit confirmation. Report the transaction receipt returned by the CLI.
