# Claude Code

Claude Code exposes the seven native commands: `/mlx-scout`, `/mlx-adopt`, `/mlx-wire`, `/mlx-bench`, `/mlx-doctor`, `/mlx-watch`, and `/mlx-fleet`.

Install from the marketplace:

```bash
claude plugin marketplace add cavi-ai/mlx-agent
claude plugin install mlx-agent
```

Verify with `claude plugin list`, restart Claude Code, and run `/mlx-scout --fast --limit 1`. Update with `claude plugin update mlx-agent`; uninstall with `claude plugin uninstall mlx-agent`.

For a receipt-owned local package, preview then confirm the universal installer:

```bash
python3 scripts/mlx-agent install claude --scope user --dry-run --json
python3 scripts/mlx-agent install claude --scope user --confirm --preview-hash <preview-hash> --json
python3 scripts/mlx-agent update claude --scope user --dry-run --json
python3 scripts/mlx-agent uninstall claude --scope user --dry-run --json
python3 scripts/mlx-agent doctor claude --scope user --json
```

Project scope uses `--scope project --project <project>`. The universal installer stages one complete self-contained package under `~/.claude/plugins/mlx-agent` or `<project>/.claude/plugins/mlx-agent`, including all seven commands, the MCP transport, and the bundled runtime. `doctor` reports staged artifacts separately from Claude-managed marketplace visibility. Marketplace-managed files remain owned by Claude Code.
