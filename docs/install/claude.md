# Claude Code

Claude Code exposes the three native commands: `/mlx-scout`, `/mlx-adopt`, and `/mlx-wire`.

Install from the marketplace:

```bash
claude plugin marketplace add sasan1200/mlx-agent
claude plugin install mlx-agent
```

For a receipt-owned local package, preview then confirm the universal installer:

```bash
python3 scripts/mlx-agent install claude --scope user --dry-run --json
python3 scripts/mlx-agent install claude --scope user --confirm --preview-hash <preview-hash> --json
python3 scripts/mlx-agent update claude --scope user --dry-run --json
python3 scripts/mlx-agent uninstall claude --scope user --dry-run --json
python3 scripts/mlx-agent doctor claude --scope user --json
```

Project scope uses `--scope project --project <project>`. The installer target roots are `~/.claude` and `<project>/.claude`; marketplace-managed files remain owned by Claude Code. Current evidence covers generated/package contracts and bundle execution, not a current live Claude CLI smoke or model response.
