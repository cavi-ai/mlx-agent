---
name: mlx-scout
description: Discover MLX-optimized models on HuggingFace suited to this Apple Silicon host, by role.
---

Discover local MLX models worth running on this Mac and help the user choose.

1. Run the discovery script, passing through any flags the user gave (`--role <r>`, `--new`, `--json`, `--limit N`):

   `python3 ${CLAUDE_PLUGIN_ROOT}/skills/mlx-scout/scripts/scout.py $ARGUMENTS`

2. Present the ranked models per role. Call out ⭐ reputable publishers and the `fits` / `reasoning` flags.
3. If the user named a role or goal, recommend one pick and show how to wire it: `ollama/<tag>`, `lmstudio/<model>`, or a native `mlx_lm` / `mlx-vlm` provider (see `${CLAUDE_PLUGIN_ROOT}/skills/mlx-scout/references/runtimes.md`).
4. The `reasoning` flag is a name heuristic — never recommend a reasoning-flagged model for a fast/cheap role without noting the trade-off, and suggest a quick test-generate to confirm.
