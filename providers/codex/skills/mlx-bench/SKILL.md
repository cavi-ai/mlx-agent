---
name: "mlx-bench"
description: "Measure a locally served MLX model without downloading it."
---

Use `$mlx-agent:mlx-bench` to invoke this installed Codex skill explicitly. Codex does not support custom `/mlx-*` slash commands.

# MLX Bench

Resolve `<plugin-root>` as the absolute directory two levels above the directory containing this SKILL.md; it holds `scripts/` and `src/`. Never resolve the bundled executable from the shell working directory.

canonical capability ID: mlx-agent.bench

Measure a model already served by a running local runtime:

`python3 <plugin-root>/scripts/mlx-agent bench run --repo <repo> --runtime <runtime> --json`

Present the returned measurements as returned. Bench must not start servers, download model weights, or change configuration; the model must already be served by an existing local runtime.
