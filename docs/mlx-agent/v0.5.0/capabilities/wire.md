# Wire

Wire renders, previews, applies, inspects, and rolls back runtime configuration as a confirmation-gated transaction.

```bash
python3 scripts/mlx-agent wire render <repo> --target mlx_lm --path ./provider.json
python3 scripts/mlx-agent wire apply  <repo> --target mlx_lm --path ./provider.json
python3 scripts/mlx-agent wire apply  <repo> --target mlx_lm --path ./provider.json --confirm --preview-hash <hash>
python3 scripts/mlx-agent wire status  <receipt>
python3 scripts/mlx-agent wire rollback <receipt> --confirm --preview-hash <hash>
```

## Targets

`ollama` (exact Modelfile), `lmstudio` / `mlx_lm` / `mlx-vlm` (JSON provider blocks), and `litellm` (bounded YAML subset). Model IDs are constrained to safe `publisher/model` form; resolved secrets are rejected before any write.

## The transaction

1. **Preview** — unified diff, secret scan of before and after, per-target `preview_hash` binding the exact bytes.
2. **Confirm** — `--confirm --preview-hash` must match the reviewed preview or the apply fails stale.
3. **Apply** — advisory locks, per-target backups, atomic replace, post-validation, optional loopback health check. Any mid-mutation failure auto-restores.
4. **Receipt** — hashes, backups, and validations recorded for status and byte-exact rollback.

Wire never writes unowned files, never resolves secrets into configs, and never touches remote endpoints — health checks are loopback-only with state-changing paths banned.
