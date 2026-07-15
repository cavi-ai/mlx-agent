---
name: mlx-adopt
description: Run the MLX adoption workflow — discover, verify by test-generate, and recommend a local model-routing config.
---

Produce a verified, decision-ready local-model routing recommendation for this host by running the bundled multi-agent workflow.

Invoke the Workflow tool with the bundled script, passing the resolved plugin root and any roles the user named:

`Workflow({ scriptPath: "${CLAUDE_PLUGIN_ROOT}/scripts/mlx-adopt.workflow.mjs", args: { pluginRoot: "${CLAUDE_PLUGIN_ROOT}", roles: [$ARGUMENTS] } })`

The workflow runs three phases:
1. **Discover** — run the mlx-scout script to list MLX candidates per role.
2. **Verify** — for each top candidate, test-generate against the local runtime (Ollama / LM Studio) to detect hidden-reasoning models and confirm they load; for models not present locally, check the HuggingFace model card instead of downloading.
3. **Recommend** — synthesize a per-role model assignment with exact wiring.

Present the workflow's final recommendation and offer to apply it to the user's agent config.

Note: this spawns multiple subagents. Only run it when the user has opted into multi-agent orchestration.
