# Worked example: a legal research pack (docs only)

This example shows the **domain-agnostic** engine applied to a legal use case. No
legal-specific code exists in the plugin; only the interview answers differ.

## Interview transcript

```
What are you building? Describe the domain in a sentence.
> On-device legal contract review with scanned PDFs

Which model roles do you need? (choose one or more) [Vision / OCR, Embeddings, Coding, Reasoning, General chat, Tool use / function calling]
> Vision / OCR, General chat

Any domain keywords to prioritize? (comma-separated, optional)
> ocr, contracts, redaction

Restrict to specific licenses? (comma-separated, optional)
> apache-2.0, mit

Memory budget in GB? (optional, e.g. 32)
> 32

Any other constraints or notes? (optional)
> must run fully offline
```

## Resulting intent

```json
{
  "domain": "On-device legal contract review with scanned PDFs",
  "roles": ["vision", "general"],
  "keywords": ["ocr", "contracts", "redaction"],
  "license_allow": ["apache-2.0", "mit"],
  "memory_gb": 32.0,
  "notes": "must run fully offline"
}
```

## Resulting pack (excerpt)

```markdown
# MLX Research Pack: On-device legal contract review with scanned PDFs

- Roles: vision, general
- Keywords: ocr, contracts, redaction
- License filter: apache-2.0, mit
- Memory budget (GB): 32.0

## Candidates

### 1. `mlx-community/<an-ocr-vision-model>` — score 82.0/100

- Role: vision
- Suggested wiring: mlx-vlm server -> ...
- Why it ranked:
  - role_match: matched roles: vision
  - keyword_match: matched keywords: ocr
  - license_ok: license: apache-2.0
  - memory_fit: est_ram_gb ... vs budget 32.0
```

The OCR emphasis comes entirely from the interview answers and card/keyword
scoring — the engine itself is domain-agnostic. Swap the answers (for example
"audio transcription" with `asr, whisper` keywords) and the same engine produces
an audio pack with no code change.
