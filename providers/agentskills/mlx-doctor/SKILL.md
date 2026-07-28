---
name: "mlx-doctor"
description: "Diagnose local model inventories, wiring drift, and endpoint health."
---

# MLX Doctor

Resolve `<skill-dir>` as the absolute directory containing this SKILL.md. Never resolve the bundled executable from the shell working directory.

canonical capability ID: mlx-agent.doctor

Run the read-only model diagnostics:

`python3 <skill-dir>/scripts/mlx-agent doctor models --json`

Present the inventory, drift findings, and endpoint health as returned. Doctor must not delete, move, or repair anything; the confirmation-gated prune of incomplete cache snapshots requires an explicit reviewed preview from the user first.
