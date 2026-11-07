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

For a fixed read-file and two-edit correctness workload, use `python3 <plugin-root>/scripts/mlx-agent bench coding --repo <repo> --runtime mlx_lm --json`. Report quality checks and cache provenance with the timings. This synthetic workload does not establish general coding quality or a comparative speed advantage.

Present the returned measurements as returned. Bench must not start servers, download model weights, or change configuration; the model must already be served by an existing local runtime.
