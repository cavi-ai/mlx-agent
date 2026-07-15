---
name: mlx-wire
description: Pull a chosen local MLX model and wire it into your agent config for a given runtime target.
---

Set up a specific local model end-to-end: fetch its setup + config, then — with confirmation — pull it and apply the config.

1. Emit the setup commands + config block for the user's chosen model and target:

   `python3 ${CLAUDE_PLUGIN_ROOT}/skills/mlx-scout/scripts/scout.py --wire <repo> --target <ollama|lmstudio|mlx_lm|mlx-vlm|litellm> [--port N]`

2. Show the user the exact pull/serve command(s) and the provider/config block.
3. **Confirm before any side effect.** Pulling downloads several GB, and applying config edits the user's setup — get an explicit yes first.
4. On confirmation: run the pull/serve command, then insert the provider block into the target config (back the config up first) and validate it parses.
5. Report what was pulled and where the config was applied. Never write tokens/keys into the config — reference an env var.

If the user hasn't chosen a model yet, run `/mlx-scout` (or `/mlx-adopt`) first.
