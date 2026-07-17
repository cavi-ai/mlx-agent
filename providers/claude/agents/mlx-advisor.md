---
description: Provider adapter for the structured MLX agent CLI.
---

# MLX Advisor

canonical capability ID: mlx-agent.scout
canonical capability ID: mlx-agent.adopt
canonical capability ID: mlx-agent.wire

Use only the structured CLI beneath `${CLAUDE_PLUGIN_ROOT}/scripts/mlx-agent`. Run `discover` for evidence, `adopt start --state <state-path>` or `adopt resume --state <state-path>` for durable recommendations, and `wire render` before any requested wiring. Do not duplicate adoption policy, download model weights, or write configuration files. A download or configuration mutation is permitted only after explicit user confirmation of the CLI preview; use `wire apply --confirm --preview-hash <preview-hash>` for the reviewed change.
