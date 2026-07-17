# Wire: preview, confirm, recover

Wire renders, previews, and applies runtime/provider configuration only. It does not pull, install, or download model weights. A separately reviewed preview hash is required before it writes configuration.

```bash
python3 scripts/mlx-agent wire render mlx-community/example --target mlx_lm --path config.yaml --json
python3 scripts/mlx-agent wire apply mlx-community/example --target mlx_lm --path config.yaml --json
python3 scripts/mlx-agent wire apply mlx-community/example --target mlx_lm --path config.yaml --confirm --preview-hash <preview-hash> --json
```

The unconfirmed `wire apply` response contains the diff and `preview_hash`. Re-render if the file changes; a stale hash is rejected. A confirmed run writes a non-secret receipt to `--receipts-dir`, or by default beside the target under `.mlx-agent-receipts/<transaction-id>/receipt.json`.

If a runtime needs a model-fetch command, that is a separate user action, not a Wire operation. For example, an Ollama user may explicitly run this outside `mlx-agent` after reviewing the model and confirming the action:

```bash
ollama pull <model>
```

Wire never executes that command or any equivalent runtime install/pull command.

Inspect and restore only the receipt you reviewed:

```bash
python3 scripts/mlx-agent wire status <receipt-path> --json
python3 scripts/mlx-agent wire rollback <receipt-path> --confirm --json
```

Wire uses target-scoped advisory locks to prevent accidental/cooperative concurrent writers. A malicious process that ignores the lock can still race the final rename; review the receipt status and target content after any recovery-required result. Use `/mlx-wire`, or `$mlx-agent:mlx-wire` in Codex.
