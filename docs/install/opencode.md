# OpenCode

OpenCode is packaged for `/mlx-scout`, `/mlx-adopt`, and `/mlx-wire`, backed by a narrow native custom tool.

Use the receipt-owned installer; it never creates or edits `opencode.json`:

```bash
python3 scripts/mlx-agent install opencode --scope user --dry-run --json
python3 scripts/mlx-agent install opencode --scope user --confirm --preview-hash <preview-hash> --json
python3 scripts/mlx-agent update opencode --scope user --dry-run --json
python3 scripts/mlx-agent uninstall opencode --scope user --dry-run --json
python3 scripts/mlx-agent doctor opencode --scope user --json
```

User-scope artifacts go to `$XDG_CONFIG_HOME/opencode` when `XDG_CONFIG_HOME` is set, otherwise `~/.config/opencode`. Project scope uses `<project>/.opencode`. The package includes commands, the `mlx-advisor` agent, the TypeScript plugin, skills, and the bundled Python runtime.

To keep an OpenCode harness on a separate disk without changing native `HOME`, use the same scoped XDG variables for installation and every OpenCode launch:

```bash
export XDG_CONFIG_HOME=/Volumes/your-disk/.config
export XDG_DATA_HOME=/Volumes/your-disk/.local/share
export XDG_STATE_HOME=/Volumes/your-disk/.local/state
export XDG_CACHE_HOME=/Volumes/your-disk/.cache

python3 scripts/mlx-agent install opencode --scope user --dry-run --json
python3 scripts/mlx-agent install opencode --scope user --confirm --preview-hash <preview-hash> --json
opencode
```

In OpenCode, press `Ctrl+P` and filter for `mlx` to verify `/mlx-scout`, `/mlx-adopt`, and `/mlx-wire`. `opencode debug paths` should report the intended XDG config, data, state, and cache roots. OpenCode 1.18.3 discovered all three commands during native validation; the model-backed provider response remained blocked by unavailable network access.
