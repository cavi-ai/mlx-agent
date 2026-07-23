# Worked example: video foundation (docs only)

This example shows the `video` foundational modality with facets
`understanding`, `generation`, and `action`. Profiles seed existing `vision`
and `general` roles — no new video discovery runtime.

## CLI

```bash
python3 scripts/mlx-agent research \
  --domain "sports highlight captioning" \
  --modality video --facet understanding
```

Or detect from domain text containing `video` / `caption`.

## Resulting intent (shape)

```json
{
  "domain": "sports highlight captioning",
  "roles": ["vision", "general"],
  "keywords": ["video", "caption", "understanding"],
  "modalities": ["video"],
  "facets": ["understanding"]
}
```

## Pack excerpt

```markdown
## Runtime preference

- Preferred: `mlx-vlm`
- Alternates: `lmstudio` when its server is up
- Rationale: Video understanding needs native vision serving; Ollama does not run VLMs.

## Modality foundations

- `video` — Video
  - Facets: `understanding` (Understanding / captioning)

## Candidates
...
```

Generation and action packs reuse the same foundation with `--facet generation`
or `--facet action`.
