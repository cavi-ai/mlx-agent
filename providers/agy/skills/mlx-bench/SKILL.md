---
name: "mlx-bench"
description: "Measure a locally served MLX model without downloading it."
---

# MLX Bench

Resolve `<skill-dir>` as the absolute directory containing this SKILL.md. Never resolve the bundled executable from the shell working directory.

canonical capability ID: mlx-agent.bench

Measure a model already served by a running local runtime:

`python3 <skill-dir>/scripts/mlx-agent bench run --repo <repo> --runtime <runtime> --json`

Present the returned measurements as returned. Bench must not start servers, download model weights, or change configuration; the model must already be served by an existing local runtime.
