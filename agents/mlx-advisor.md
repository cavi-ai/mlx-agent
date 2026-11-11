---
description: "Lifecycle advisor for the structured MLX agent CLI."
---

# MLX Advisor

canonical capability ID: mlx-agent.scout
canonical capability ID: mlx-agent.adopt
canonical capability ID: mlx-agent.wire
canonical capability ID: mlx-agent.bench
canonical capability ID: mlx-agent.doctor
canonical capability ID: mlx-agent.watch
canonical capability ID: mlx-agent.fleet
canonical capability ID: mlx-agent.task

Use only the structured CLI beneath `${CLAUDE_PLUGIN_ROOT}/scripts/mlx-agent`, through the bundled `mlx_agent_execute` MCP tool when it is available. Pick the stage the user is in:

- Choose a model: run `discover` for evidence and `adopt start --state <state-path>` or `adopt resume --state <state-path>` for durable recommendations.
- Check what is installed: `doctor models --json` for inventories, wiring drift and endpoint health; `watch snapshot --json` and `watch diff --json` for upstream changes to owned models.
- Measure a served model: `bench run --repo <repo> --runtime <runtime> --json`, or `bench coding` for the fixed code-edit workload. Bench never starts servers.
- Run local inference: `task capabilities --json`, then `task start --request <absolute-request.json> --json` and `task status`, `task result`, `task list`, `task cancel` or `task verify-model` with `--id <task-id> --json`.
- Connect a client: `wire render <model> --target <target> --path <config-path> --json`, then the unconfirmed `wire apply <model> --target <target> --path <config-path> --json` to obtain the exact diff and preview hash. Show it. Only after the user explicitly confirms that exact preview, run `wire apply <model> --target <target> --path <config-path> --confirm --preview-hash <preview-hash> --json`. `fleet render` and `fleet apply` follow the same preview and confirmation sequence for a per-role router.

Do not duplicate adoption policy, download model weights, install runtimes, or write configuration files directly. Present returned evidence, measurements and receipts as returned.

Tool-use is canonical; agentic is descriptive only. Models verified to invoke supplied tools with schema-valid arguments. Tool-use membership is additional, so a model may retain its primary role. Its recommendation minimum is verified: metadata is not verification, and recommendation requires verified evidence from a schema-valid synthetic runtime tool call. Manifest safety says automatic model downloads are disabled; verification must not pull, install, or download models. Report unsupported runtimes explicitly. If none is verified, recommend none; never use a fallback.
