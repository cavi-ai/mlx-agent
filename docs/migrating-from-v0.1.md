# Migrate from the v0.1 Claude marketplace package

Existing Claude users can keep the marketplace installation and continue using `/mlx-scout`, `/mlx-adopt`, and `/mlx-wire`. The universal package adds provider-specific layouts and receipt-owned install/update/uninstall workflows; it does not ask you to hand-edit a Claude marketplace file.

Before changing an existing installation, inspect it and create a no-write plan:

```bash
claude plugin marketplace add sasan1200/mlx-agent
claude plugin install mlx-agent
python3 scripts/mlx-agent install claude --scope user --dry-run --json
```

If the plan reports an unowned or modified artifact, keep the existing package in place and resolve its ownership before confirming anything. Do not delete it to force the installer through. For a receipt-owned universal installation, pass the reviewed hash to the confirmation command, then validate with `doctor`:

```bash
python3 scripts/mlx-agent install claude --scope user --confirm --preview-hash <preview-hash> --json
python3 scripts/mlx-agent doctor claude --scope user --json
```

To remove a receipt-owned universal package, first inspect `uninstall --dry-run`, then confirm its exact hash. Marketplace-managed content should be removed through Claude Code's plugin controls, not through an MLX receipt that does not own it.
