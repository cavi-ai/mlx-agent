# Research packs

`mlx-agent research` turns a short domain description into a ranked, evidence-backed
research pack written as project-local markdown. It is **read-only**: it never
verifies, wires, or downloads a model. Use it to dial in a local stack for a
domain, then hand the pack to another agent or to `mlx-agent adopt`.

## Quick start

```bash
# Non-interactive
python3 scripts/mlx-agent research --domain "legal contract review" --role vision --keyword ocr

# Interactive interview
python3 scripts/mlx-agent research --interview
```

The pack is written to `./mlx-research/<domain-slug>-<timestamp>.md`. Pass
`--project DIR` to change the project root, `--json` for machine output, or
`--no-write` to render without writing a file.

## What the interview asks

The interview is a deterministic template (domain, roles, keywords, license
filter, memory budget, notes). Answers become a validated `DomainIntent`. An
optional assist layer may refine answers, but every assist result is re-validated
through the same deterministic path and can never inject an unknown role or bypass
validation.

## How candidates are scored

Each candidate is scored 0–100 from transparent signals with per-signal
provenance: role match, keyword match (model card text + tags), popularity,
license fit, memory fit, and model-card quality. Signals you did not ask for
(for example a license filter you left blank, or a memory budget you did not set)
are excluded from the score, not penalized. All scores are estimates; verify
capability with `mlx-agent adopt` before relying on any model.

## Safety

Research fetches only bounded model metadata and README/model-card text over a
fixed HTTPS host, redirect- and proxy-hardened, with a strict byte cap. It writes
only inside `<project>/mlx-research` and refuses paths that resolve outside that
folder or through a symlink. No model is downloaded, verified, or wired.
