---
description: "Provider adapter for the structured MLX agent CLI."
---

# MLX Advisor

canonical capability ID: mlx-agent.scout
canonical capability ID: mlx-agent.adopt
canonical capability ID: mlx-agent.wire

Use only the structured CLI beneath `${CLAUDE_PLUGIN_ROOT}/scripts/mlx-agent`. Run `discover` for evidence and `adopt start --state <state-path>` or `adopt resume --state <state-path>` for durable recommendations. For wiring, run `wire render <model> --target <target> --path <config-path> --json`, then the unconfirmed `wire apply <model> --target <target> --path <config-path> --json` to obtain the exact diff and preview hash. Show it. Only after the user explicitly confirms that exact preview, run `wire apply <model> --target <target> --path <config-path> --confirm --preview-hash <preview-hash> --json`. Do not duplicate adoption policy, download model weights, or write configuration files.
