# OpenCode

OpenCode is packaged for `/mlx-scout`, `/mlx-adopt`, `/mlx-wire`, `/mlx-bench`, `/mlx-doctor`, `/mlx-watch`, `/mlx-fleet`, and `/mlx-task`, backed by a narrow native custom tool.

Both packaged plugins require OpenCode 2.3.2 or later and the OpenCode 2 plugin API (`@opencode/plugin`).

Use the receipt-owned installer; it never creates or edits `opencode.json`:

```bash
python3 scripts/mlx-agent install opencode --scope user --dry-run --json
python3 scripts/mlx-agent install opencode --scope user --confirm --preview-hash <preview-hash> --json
python3 scripts/mlx-agent update opencode --scope user --dry-run --json
python3 scripts/mlx-agent uninstall opencode --scope user --dry-run --json
python3 scripts/mlx-agent doctor opencode --scope user --json
```

User-scope artifacts go to `$XDG_CONFIG_HOME/opencode` when `XDG_CONFIG_HOME` is set, otherwise `~/.config/opencode`. Project scope uses `<project>/.opencode`. The package includes commands, the `mlx-advisor` agent, two OpenCode 2 plugins (`mlx-agent-command.ts` and `mlx-agent-provider.ts`), skills, and the Python runtime (`src/mlx_agent`, copied from the repository root).

## Model provider

`mlx-agent-provider.ts` registers provider `mlx-agent` when OpenCode starts. Every live chat endpoint that `serve discover` finds (`mlx_lm`, `mlx-vlm`, `strata`, `splash`) becomes model `mlx-agent/<model name>`, using the path-free name the front door lists. A keyed endpoint is included when its receipt names an API key file this user can read; the key stays in memory and is never written to `opencode.json`. A name served by two endpoints is left out. By default, the provider connects directly to each listed endpoint. A shared front door uses its own URL and key.

```bash
opencode models | grep '^mlx-agent/'
opencode run -m "mlx-agent/<model name>" "Reply with OK."
```

Endpoints started after OpenCode are listed after OpenCode restarts.

To use a shared authenticated front door, set `MLX_AGENT_FRONT_DOOR_URL` to `http://127.0.0.1:<port>/v1` and provide its key in `MLX_AGENT_FRONT_DOOR_KEY` in the OpenCode process environment. Both values are required. The provider authenticates the front door and lists its chat routes; task routes are excluded. If either value is invalid or authentication fails, no direct endpoints are listed. The key is passed in memory and is not written to `opencode.json`.

A symlinked config directory needs no configuration. If `~/.config/opencode` points at another location, the installer resolves it once and writes to the real directory; the resolved path is what the preview and the receipt record.

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

In OpenCode, press `Ctrl+P` and filter for `mlx` to verify `/mlx-scout`, `/mlx-adopt`, `/mlx-wire`, `/mlx-bench`, `/mlx-doctor`, `/mlx-watch`, `/mlx-fleet`, and `/mlx-task`. `opencode debug paths` should report the intended XDG config, data, state, and cache roots.
