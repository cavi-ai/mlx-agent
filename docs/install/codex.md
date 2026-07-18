# Codex CLI

Codex uses installed skills, not custom slash commands. The exact native invocations are `$mlx-agent:mlx-scout`, `$mlx-agent:mlx-adopt`, and `$mlx-agent:mlx-wire` — do not use `/mlx-scout` in Codex.

Stage the local package with the universal installer if needed:

```bash
python3 scripts/mlx-agent install codex --scope user --dry-run --json
python3 scripts/mlx-agent install codex --scope user --confirm --preview-hash <preview-hash> --json
python3 scripts/mlx-agent update codex --scope user --dry-run --json
python3 scripts/mlx-agent uninstall codex --scope user --dry-run --json
python3 scripts/mlx-agent doctor codex --scope user --json
```

The staged package is under `~/plugins/mlx-agent` (or `<project>/plugins/mlx-agent`). Register and install it through the Codex marketplace CLI; Codex owns its marketplace configuration rather than the MLX installer editing it. The marketplace files are `~/.agents/plugins/marketplace.json` and `<project>/.agents/plugins/marketplace.json`.

Codex CLI 0.137.0 completed isolated marketplace registration/install and exposed all three namespaced skills in prompt injection. Its final model response was DNS-blocked, so this document makes no model-backed invocation claim.
