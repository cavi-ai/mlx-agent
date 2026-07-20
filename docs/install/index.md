# Install mlx-agent

Choose the host that owns your coding-agent surface: [Claude Code](claude.md), [Codex CLI](codex.md), [Gemini CLI](gemini.md), or [OpenCode](opencode.md). For any AgentSkills-compatible host, copy one generated `providers/agentskills/mlx-*` directory into that host's skills directory.

For a user-scoped portable install, copy all three generated packages into the host's AgentSkills directory:

```bash
mkdir -p ~/.agents/skills
cp -R providers/agentskills/mlx-scout providers/agentskills/mlx-adopt providers/agentskills/mlx-wire ~/.agents/skills/
```

For project scope, use `<project>/.agents/skills` instead. Restart the host and verify its available-skills list contains `mlx-scout`, `mlx-adopt`, and `mlx-wire`. Update by replacing only those three directories; uninstall by removing only those three package directories.

All provider packages contain the same structured Python core and require Python 3.9 or later. The universal installer stages only receipt-owned artifacts; it never installs a provider CLI, downloads model weights, persists secrets, or edits an unowned configuration file. `MLX_AGENT_CONFIG_ROOT` explicitly relocates MLX-agent receipts. When it is unset, `XDG_STATE_HOME` relocates those receipts. OpenCode additionally follows `XDG_CONFIG_HOME`; other provider user roots remain anchored to the selected host's home directory.

```bash
# Run from this repository or an unpacked release.
python3 scripts/mlx-agent providers --json
python3 scripts/mlx-agent install gemini --scope user --dry-run --json
```

Inspect the returned `preview.preview_hash`. Only then repeat the operation with the exact hash:

```bash
python3 scripts/mlx-agent install gemini --scope user --confirm --preview-hash <preview-hash> --json
```

Use the same preview/confirmation sequence for `update` and `uninstall`. `doctor` is read-only and reports `portable`, `staged`, or `native-visible` integration separately from receipt-owned artifact validity:

```bash
python3 scripts/mlx-agent update gemini --scope user --dry-run --json
python3 scripts/mlx-agent uninstall gemini --scope user --dry-run --json
python3 scripts/mlx-agent doctor gemini --scope user --json
```

For project scope, add `--scope project --project /absolute/project/path`. Project receipts stay under `<project>/.mlx-agent/installer-receipts`.
