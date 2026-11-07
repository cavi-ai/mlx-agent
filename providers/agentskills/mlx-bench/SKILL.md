---
name: "mlx-bench"
description: "Measure a locally served MLX model without downloading it."
---

# MLX Bench

Resolve `<skill-dir>` as the absolute directory containing this SKILL.md. Never resolve the bundled executable from the shell working directory.

canonical capability ID: mlx-agent.bench

Measure a model already served by a running local runtime:

`python3 <skill-dir>/scripts/mlx-agent bench run --repo <repo> --runtime <runtime> --json`

For a fixed read-file and two-edit correctness workload, use `python3 <skill-dir>/scripts/mlx-agent bench coding --repo <repo> --runtime mlx_lm --json`. Report quality checks and cache provenance with the timings. This synthetic workload does not establish general coding quality or a comparative speed advantage.

Present the returned measurements as returned. Bench must not start servers, download model weights, or change configuration; the model must already be served by an existing local runtime.
