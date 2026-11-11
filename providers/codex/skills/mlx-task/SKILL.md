---
name: "mlx-task"
description: "Run, track and collect durable local inference tasks."
---

Use `$mlx-agent:mlx-task` to invoke this installed Codex skill explicitly. Codex does not support custom `/mlx-*` slash commands.

# MLX Task

Resolve `<plugin-root>` as the absolute directory two levels above the directory containing this SKILL.md; it holds `scripts/` and `src/`. Never resolve the bundled executable from the shell working directory.

canonical capability ID: mlx-agent.task

List the supported inference kinds and their request contracts without loading a model:

`python3 <plugin-root>/scripts/mlx-agent task capabilities --json`

Add `--model-path <absolute-model-dir>` to read one local model's optional media settings. To run a task, write a `task-request` JSON file whose `kind`, `model_path` and `parameters` follow the returned contract, then start it with an absolute path:

`python3 <plugin-root>/scripts/mlx-agent task start --request <absolute-request.json> --json`

Follow it with `task status --id <task-id> --json`, collect verified output paths with `task result --id <task-id> --json`, and review recent runs with `task list --json`. `task cancel --id <task-id> --json` stops a running task; `task verify-model --id <task-id> --json` rechecks that the model still matches the one the task used.

Present the returned status, metrics and artifact paths as returned. Tasks use only an existing local model and its installed backend; they must not download model weights, install a runtime, or change configuration.
