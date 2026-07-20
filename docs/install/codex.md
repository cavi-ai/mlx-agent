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

For the native Codex plugin lifecycle, register this checkout as a local marketplace and install the package:

```bash
codex plugin marketplace add "$PWD"
codex plugin add mlx-agent@mlx-agent
codex plugin list
```

Restart Codex, then invoke `$mlx-agent:mlx-scout`, `$mlx-agent:mlx-adopt`, or `$mlx-agent:mlx-wire`. Update the checkout and run `codex plugin marketplace upgrade mlx-agent`; remove the package with `codex plugin remove mlx-agent@mlx-agent` and, if desired, `codex plugin marketplace remove mlx-agent`.

The universal installer stages the package under `~/plugins/mlx-agent` (or `<project>/plugins/mlx-agent`) but deliberately does not edit Codex-owned marketplace configuration. The repository-level `.agents/plugins/marketplace.json` is the installable local marketplace entrypoint.
