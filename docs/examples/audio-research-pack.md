# Worked example: audio foundation (docs only)

This example shows the `audio` foundational modality (ASR / TTS / music). No
audio-specific discovery roles are added; the profile seeds the existing
`general` role plus ASR keywords.

## CLI

```bash
python3 scripts/mlx-agent research \
  --domain "meeting transcription and summaries" \
  --modality audio --facet asr \
  --keyword whisper
```

Or rely on detection from domain text (`transcription` / `whisper`).

## Resulting intent (shape)

```json
{
  "domain": "meeting transcription and summaries",
  "roles": ["general"],
  "keywords": ["whisper", "asr", "transcription", "speech-to-text", "audio", "speech"],
  "modalities": ["audio"],
  "facets": ["asr"]
}
```

## Pack excerpt

```markdown
## Runtime preference

- Preferred: `lmstudio` (or `mlx_lm` when LM Studio is down)
- Alternates: `ollama` when detected
- Rationale: Audio packs prefer native MLX; Ollama is limited to curated tags.

## Modality foundations

- `audio` — Audio (ASR / TTS / music)
  - Facets: `asr` (ASR / speech-to-text)

## Candidates
...
```

Music or TTS packs use the same foundation with `--facet music` or `--facet tts`.
