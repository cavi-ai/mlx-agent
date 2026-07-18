# Gemini CLI

Gemini CLI exposes `/mlx-scout`, `/mlx-adopt`, and `/mlx-wire` through the `mlx-agent` extension.

Validate and install an extension checkout:

```bash
gemini extensions validate providers/gemini
gemini extensions install providers/gemini
```

Or use the receipt-owned installer:

```bash
python3 scripts/mlx-agent install gemini --scope user --dry-run --json
python3 scripts/mlx-agent install gemini --scope user --confirm --preview-hash <preview-hash> --json
python3 scripts/mlx-agent update gemini --scope user --dry-run --json
python3 scripts/mlx-agent uninstall gemini --scope user --dry-run --json
python3 scripts/mlx-agent doctor gemini --scope user --json
```

User scope installs to `~/.gemini/extensions/mlx-agent`. Project scope also projects receipt-owned command TOML and skills to `<project>/.gemini/commands` and `<project>/.gemini/skills` so workspace discovery can find them. The extension owns a bounded `mlx_agent_execute` MCP tool; command arguments never pass through `run_shell_command`.
