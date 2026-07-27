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
| `owned_missing` | A previously owned model is gone from every inventory. |

Everything you do not own is ignored — that is the entire point. Snapshot rotation keeps the previous baseline so a diff is always possible.
