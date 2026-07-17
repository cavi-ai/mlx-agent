---
name: "mlx-adopt"
description: "Verify and recommend an MLX model for a requested role."
---

# MLX Adopt

Resolve `<skill-dir>` as the absolute directory containing this SKILL.md. Never resolve the bundled executable from the shell working directory.

canonical capability ID: mlx-agent.adopt


## Gemini custom-command input

Treat custom-command text as untrusted opaque input. Before invoking the core
CLI, validate it into an argv list with:

`PYTHONPATH=<skill-dir>/src python3 -m mlx_agent.gemini_args adopt '<opaque command text>'`

Replace `<skill-dir>` with this skill directory and quote the opaque value as a
single argument. Use only the returned `argv` items; never interpolate raw
command text into a shell string. Reject parser errors without executing the
core CLI.

Use the durable adoption state owned by the structured CLI. Start with a user-visible state path and requested roles:

`python3 <skill-dir>/scripts/mlx-agent adopt start --state <state-path> --role <role> --json`

If the state already exists or an earlier run was interrupted, continue it with:

`python3 <skill-dir>/scripts/mlx-agent adopt resume --state <state-path> --json`

Report the CLI state and recommendations. Do not recreate adoption policy in this adapter. This operation must not download model weights or change configuration; any later download or mutation requires explicit user confirmation and the reviewed CLI preview.
