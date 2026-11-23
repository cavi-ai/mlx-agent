---
name: "mlx-doctor"
description: "Diagnose local model inventories, wiring drift, and endpoint health."
---

# MLX Doctor

Resolve `<plugin-root>` as the absolute directory two levels above the directory containing this SKILL.md; it holds `scripts/` and `src/`. Never resolve the bundled executable from the shell working directory.

canonical capability ID: mlx-agent.doctor

Run the read-only model diagnostics:

`python3 <plugin-root>/scripts/mlx-agent doctor models --json`

Present the inventory, drift findings, and endpoint health as returned. Doctor must not delete, move, or repair anything; the confirmation-gated prune of incomplete cache snapshots requires an explicit reviewed preview from the user first.
