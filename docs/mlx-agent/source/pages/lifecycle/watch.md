# Watch

Watch is a stateful digest of what changed on Hugging Face for models you own.

```bash
python3 scripts/mlx-agent watch snapshot   # record a baseline
python3 scripts/mlx-agent watch diff       # what changed since, for models you own
```

`discover --new` only re-sorts the Hub by lastModified. Watch instead persists owned inventories (HF cache, runtimes, wired configs) and a full-role discovery reading into one self-owned state file, then classifies only owned-relevant changes:

| Code | Meaning |
| --- | --- |
| `new_quant_of_owned` | A new repository matching an owned model's base. |
| `updated_tracked_repo` | Weight bytes changed on a tracked repo. |
| `gated_changed` | Gated flag flipped on a tracked repo. |
| `owned_missing` | A previously owned model is absent from checked inventories within an unchanged scope. |

Everything you do not own is ignored — that is the entire point. Snapshot rotation keeps the previous baseline so a diff is always possible.

## Choose the ownership scope

Repeat `--wired-root` to inspect only selected client files or directories. Omit it to inspect the current project. `--hf-cache PATH` selects one cache; `--no-hf-cache` excludes cache inspection. These two flags are mutually exclusive. Already-running loopback runtimes are still queried.

```bash
python3 scripts/mlx-agent watch snapshot --state-dir ./watch-state \
  --wired-root ./client.json --wired-root ./receipts \
  --no-hf-cache --owned-repo mlx-community/Qwen3-8B-4bit
python3 scripts/mlx-agent watch diff --state-dir ./watch-state \
  --wired-root ./client.json --wired-root ./receipts \
  --no-hf-cache --owned-repo mlx-community/Qwen3-8B-4bit
```

Repeat `--owned-repo org/name` for models installed outside a Hugging Face cache. These IDs declare ownership without searching their folders. URLs, local paths and checkpoint paths are refused.

Snapshots retain the selected `scope` and `ownership_checks` as additive fields in schema `1.0`. JSON results include the same fields. Changed scope, excluded sources, unavailable runtimes and incomplete inventory evidence suppress `owned_missing`; upstream weight-size, access and related-upload findings remain available. Wired scans prove presence but can skip unreadable or invalid files, so they do not establish absence. Record a new snapshot after changing scope to establish a matching baseline. Baselines saved before scope metadata was available remain readable and do not establish absence.

`serve discover --json` advertises `capabilities.scoped_watch: true` when these scope flags are supported. Clients that require scoped reads must check this capability before invoking Watch.
