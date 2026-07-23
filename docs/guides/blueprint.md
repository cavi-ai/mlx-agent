# Project design packs

`mlx-agent blueprint` turns a short project goal into a guidance-only MLX
**project design pack** (markdown + JSON). Use it so downstream agents can plan
quantization, training loops, LoRA, and study materials before any scaffolding
or training.

It does **not** scaffold repositories, download weights, or run training.

## Quick start

```bash
python3 scripts/mlx-agent blueprint \
  --goal "On-device legal OCR assistant" \
  --modality document-vision \
  --memory-gb 64

python3 scripts/mlx-agent blueprint \
  --goal "Meeting ASR notes" \
  --modality audio \
  --notes "must stay offline"

# Render only (no files)
python3 scripts/mlx-agent blueprint --goal "Billing helper" --no-write
```

Packs land under `./mlx-blueprints/<goal-slug>-<timestamp>.md` with a matching
`.json` sidecar. Pass `--project DIR` to change the project root, or `--json`
for machine output.

## Brief fields

| Flag | Purpose |
|---|---|
| `--goal` | Required one-line project goal |
| `--modality` | Repeatable foundation: `audio`, `video`, `document-vision` |
| `--memory-gb` | Optional host memory budget (shapes quant guidance) |
| `--notes` | Free-form constraints echoed into next steps |

## Pack sections

- Goal and brief metadata
- Recommended stack path (`research` → runtime preference → `adopt` → `wire`)
- Quantization ideas (modality- and memory-aware)
- Training loop sketch (dry-run / LoRA-first; no execution)
- LoRA / adapter notes
- Experimental MTX notes (optional research, not production defaults)
- Study materials
- Next steps

## Safety

Writes only inside `<project>/mlx-blueprints` and refuses symlink escape. No
Hub fetches, downloads, or training. Distinct from the dataset blueprint section
inside research packs (`blueprint.py`).
