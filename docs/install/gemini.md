# Gemini CLI

Gemini CLI loads a native extension from a directory containing `gemini-extension.json` and its `skills/` subdirectory. The universal installer copies the extension manifest, all eight skills, and one shared copy of the `mlx-agent` launcher and Python runtime into the extension root.

The extension exposes the `mlx-scout skill`, `mlx-adopt skill`, `mlx-wire skill`, `mlx-bench skill`, `mlx-doctor skill`, `mlx-watch skill`, `mlx-fleet skill`, and `mlx-task skill`.

For receipt-owned, reviewable installs, use the universal installer:

```bash
python3 scripts/mlx-agent install gemini --scope user --dry-run --json
python3 scripts/mlx-agent install gemini --scope user --confirm --preview-hash <preview-hash> --json
python3 scripts/mlx-agent update gemini --scope user --dry-run --json
python3 scripts/mlx-agent uninstall gemini --scope user --dry-run --json
python3 scripts/mlx-agent doctor gemini --scope user --json
```

The user extension root is `~/.gemini/extensions/mlx-agent`; the project extension root is `<project>/.gemini/extensions/mlx-agent`. For project scope, pass `--scope project --project /absolute/project/path`. Inspect the returned `--preview-hash` before any mutating `install`, `update`, or `uninstall` call.
