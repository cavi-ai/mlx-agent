---
name: "mlx-task"
description: "Run, track and collect durable local inference tasks."
---

# MLX Task

Resolve `<skill-dir>` as the absolute directory containing this SKILL.md. Never resolve the bundled executable from the shell working directory.

canonical capability ID: mlx-agent.task

List the supported inference kinds and their request contracts without loading a model:

`python3 <skill-dir>/scripts/mlx-agent task capabilities --json`

Add `--model-path <absolute-model-dir>` to read one local model's optional media settings. To run a task, write a `task-request` JSON file whose `kind`, `model_path` and `parameters` follow the returned contract, then start it with an absolute path:

`python3 <skill-dir>/scripts/mlx-agent task start --request <absolute-request.json> --json`

Follow it with `task status --id <task-id> --json`, collect verified output paths with `task result --id <task-id> --json`, and review recent runs with `task list --json`. `task cancel --id <task-id> --json` stops a running task; `task verify-model --id <task-id> --json` rechecks that the model still matches the one the task used.

Present the returned status, metrics and artifact paths as returned. Tasks use only an existing local model and its installed backend; they must not download model weights, install a runtime, or change configuration.
