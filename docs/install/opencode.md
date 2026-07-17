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

Artifacts go to `~/.config/opencode` for user scope and `<project>/.opencode` for project scope. They include commands, the `mlx-advisor` agent, the TypeScript plugin, skills, and the bundled Python runtime.

OpenCode and Bun were unavailable during validation. Static package contracts and an equivalent no-shell argv/stdin fixture passed, but the shipped TypeScript plugin, native discovery, and a model-backed invocation were not run.
