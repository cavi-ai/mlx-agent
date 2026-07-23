---
description: "Provider adapter for the structured MLX agent CLI."
---

# MLX Advisor

canonical capability ID: mlx-agent.scout
canonical capability ID: mlx-agent.adopt
canonical capability ID: mlx-agent.wire

Use only the structured CLI beneath `${CLAUDE_PLUGIN_ROOT}/scripts/mlx-agent`. Run `discover` for evidence and `adopt start --state <state-path>` or `adopt resume --state <state-path>` for durable recommendations. For wiring, run `wire render <model> --target <target> --path <config-path> --json`, then the unconfirmed `wire apply <model> --target <target> --path <config-path> --json` to obtain the exact diff and preview hash. Show it. Only after the user explicitly confirms that exact preview, run `wire apply <model> --target <target> --path <config-path> --confirm --preview-hash <preview-hash> --json`. Do not duplicate adoption policy, download model weights, or write configuration files.

Tool-use is canonical; agentic is descriptive only. Models verified to invoke supplied tools with schema-valid arguments. Tool-use membership is additional, so a model may retain its primary role. Its recommendation minimum is verified: metadata is not verification, and recommendation requires verified evidence from a schema-valid synthetic runtime tool call. Manifest safety says automatic model downloads are disabled; verification must not pull, install, or download models. Report unsupported runtimes explicitly. If none is verified, recommend none; never use a fallback.
