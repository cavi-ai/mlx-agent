# Install mlx-agent

Choose the host that owns your coding-agent surface: [Claude Code](claude.md), [Codex CLI](codex.md), [Gemini CLI](gemini.md), or [OpenCode](opencode.md). For any AgentSkills-compatible host, copy one generated `providers/agentskills/mlx-*` directory into that host's skills directory.

All provider packages contain the same structured Python core and require Python 3.9 or later. The universal installer stages only receipt-owned artifacts; it never installs a provider CLI, downloads model weights, persists secrets, or edits an unowned configuration file. `MLX_AGENT_CONFIG_ROOT` relocates MLX-agent receipts/state only; provider user roots remain anchored to the selected host's home directory.

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
