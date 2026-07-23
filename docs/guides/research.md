# Research packs

`mlx-agent research` turns a short domain description into a ranked, evidence-backed
research pack written as project-local markdown (and a JSON sidecar). It is
**read-only**: it never verifies, wires, or downloads a model, adapter, or
dataset. Use it to dial in a local stack for a domain, then hand the pack to
another agent or to `mlx-agent adopt`.

## Quick start

```bash
# Non-interactive (modality detected from domain/keywords, or pass --modality)
python3 scripts/mlx-agent research --domain "legal contract review" --role vision --keyword ocr

# Explicit foundational modality
python3 scripts/mlx-agent research --domain "billing helper" --modality audio --facet asr

# Interactive interview (detects modalities, or asks if none match)
python3 scripts/mlx-agent research --interview
```

The pack is written to `./mlx-research/<domain-slug>-<timestamp>.md` with a
matching `.json` sidecar. Pass `--project DIR` to change the project root,
`--json` for machine output, or `--no-write` to render without writing a file.

## Foundational modalities

Research seeds intent from three foundational layers (no new discovery roles):

| Foundation | Facets | Default roles seeded |
|---|---|---|
| `audio` | `asr`, `tts`, `music` | `general` |
| `video` | `understanding`, `generation`, `action` | `vision`, `general` |
| `document-vision` | `ocr`, `layout`, `general-vision` | `vision` |

Activation order:

1. CLI `--modality` / `--facet` (repeatable) if provided
2. Else deterministic lexicon detect from `--domain` and `--keyword` text
3. Else interview asks explicitly; non-interactive fails with `modality_required`

Specialized domains (legal, etc.) compose these foundations; they are not separate engines.

## What the interview asks

The interview is a deterministic template (domain, optional modality/facet
follow-ups, roles, keywords, license filter, memory budget, notes). Answers
become a validated `DomainIntent`. An optional assist layer may refine answers,
but every assist result is re-validated through the same deterministic path and
can never inject an unknown role or bypass validation.

## How candidates are scored

Each candidate is scored 0–100 from transparent signals with per-signal
provenance: role match, keyword match (model card text + tags), popularity,
license fit, memory fit, and model-card quality. Signals you did not ask for
(for example a license filter you left blank, or a memory budget you did not set)
are excluded from the score, not penalized. All scores are estimates; verify
capability with `mlx-agent adopt` before relying on any model.

## Adapters, datasets, and blueprints

After model ranking, research also searches the Hugging Face Hub (bounded,
read-only) for PEFT/LoRA adapters and datasets using the same intent keywords.
Hub rows are mapped into the same scoring contract (memory fit is N/A without an
estimate) and ranked under stable headings:

- `## Modality foundations`
- `## Adapters / LoRAs`
- `## Datasets`
- `## Dataset blueprint` — emitted only when no datasets ranked; a deterministic
  guidance template (goal, schema, labeling, splits, license/privacy, MLX next
  steps). It does not create, download, or train data.

## Safety

Research fetches only bounded Hub metadata and README/card text over a fixed
HTTPS host (models and datasets APIs), redirect- and proxy-hardened, with a
strict byte cap. It writes only inside `<project>/mlx-research` and refuses paths
that resolve outside that folder or through a symlink. No model, adapter, or
dataset is downloaded, verified, or wired.

## Worked examples

- [Legal as document-vision composition](../examples/legal-research-pack.md)
- [Audio foundation (ASR / TTS / music)](../examples/audio-research-pack.md)
- [Video foundation (understanding / generation / action)](../examples/video-research-pack.md)
