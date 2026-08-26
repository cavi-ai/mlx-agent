# Agy

Agy loads a plugin from a directory containing `plugin.json` and its `skills/`
subdirectory. Install the generated native package with Agy's lifecycle:

```bash
agy plugin validate providers/agy
agy plugin install providers/agy
agy plugin list
```

The package exposes the `mlx-scout skill`, `mlx-adopt skill`, `mlx-wire skill`,
`mlx-bench skill`, `mlx-doctor skill`, `mlx-watch skill`, and `mlx-fleet skill`.
Use `/skills` in an Agy session to inspect the loaded skills, then ask Agy to use
the named skill. Agy does not document `/mlx-*` custom slash commands for
plugin skills, so this package does not claim them.

For receipt-owned, reviewable installs, use the universal installer:

```bash
python3 scripts/mlx-agent install agy --scope user --dry-run --json
python3 scripts/mlx-agent install agy --scope user --confirm --preview-hash <preview-hash> --json
python3 scripts/mlx-agent update agy --scope user --dry-run --json
python3 scripts/mlx-agent uninstall agy --scope user --dry-run --json
python3 scripts/mlx-agent doctor agy --scope user --json
```

The user package root is `~/.gemini/config/plugins/mlx-agent`; the project
package root is `<project>/.agents/plugins/mlx-agent`. Validate the generated
package before installation, and use the preview hash before any installer
mutation.
