# Doctor

Doctor answers "what do I have, what's broken, what's wasting disk" — read-only except for the explicitly confirmed prune.

```bash
python3 scripts/mlx-agent doctor models
python3 scripts/mlx-agent doctor models --wired-root . --json
```

## Checks

- **Inventory** — Hugging Face cache (sizes, revisions, incomplete snapshots) plus Ollama and LM Studio inventories.
- **Drift** — wired configs referencing missing models (`drift_missing_model`), wired files changed since their receipt (`drift_hash_mismatch`), missing wired files, two runtimes claiming one port (`drift_endpoint_conflict`).
- **Endpoint health** — each wired loopback endpoint classified `ok`, `reachable`, or `down`.

## Prune

```bash
python3 scripts/mlx-agent doctor models --prune
python3 scripts/mlx-agent doctor models --prune --confirm --preview-hash <hash>
```

Prune deletes only incomplete cache snapshots. The preview names every candidate directory and its size and marks the deletion as irreversible; execution requires the matching preview hash and re-verifies containment inside the cache at delete time.
