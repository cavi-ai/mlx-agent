# Wire

Wire renders, previews, applies, inspects, and rolls back runtime configuration as a confirmation-gated transaction.

```bash
python3 scripts/mlx-agent wire render <repo> --target mlx_lm --path ./provider.json
python3 scripts/mlx-agent wire apply  <repo> --target mlx_lm --path ./provider.json
python3 scripts/mlx-agent wire apply  <repo> --target mlx_lm --path ./provider.json --confirm --preview-hash <hash>
python3 scripts/mlx-agent wire status  <receipt>
```

## Targets

`ollama` (exact Modelfile), `lmstudio` / `mlx_lm` / `mlx-vlm` (JSON provider blocks), and `litellm` (bounded YAML subset). Model IDs are constrained to safe `publisher/model` form; resolved secrets are rejected before any write.

## The transaction

1. **Preview** — unified diff, secret scan of before and after, per-target `preview_hash` binding the exact bytes.
2. **Confirm** — `--confirm --preview-hash` must match the reviewed preview or the apply fails stale.
3. **Apply** — advisory locks, per-target backups, atomic replace, post-validation, optional loopback health check. A mid-mutation failure attempts restoration and records any recovery requirement.
4. **Receipt** — hashes, backups, and validations recorded for status and byte-exact rollback.

Choose a dedicated target path and review the whole diff: Wire writes the
rendered configuration at that path. Provider installer ownership checks are a
separate boundary. Wire rejects resolved secrets; its health checks are
loopback-only with state-changing paths banned.

## Recover a configuration

Apply returns a receipt path. Its default location is `.mlx-agent-receipts`
beside the target, unless `--receipts-dir` is supplied. Inspect it before rollback:

```bash
python3 scripts/mlx-agent wire status <receipt> --json
python3 scripts/mlx-agent wire rollback <receipt> --json
# Review current state; use this rollback preview's data.preview.preview_hash.
python3 scripts/mlx-agent wire rollback <receipt> --confirm --preview-hash <rollback-hash> --json
```

Rollback checks current bytes against the receipt and restores the backup.
Do not reuse the apply hash. Preserve backups and inspect receipt validations
if recovery fails. Advisory locks coordinate cooperative writers; permissions
are needed against processes that ignore them. See the
[security model](../security/model.md) and [Troubleshooting](../guides/troubleshooting.md).
