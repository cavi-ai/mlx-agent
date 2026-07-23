# Worked example: legal as document-vision composition (docs only)

This example shows the **domain-agnostic** engine applied to a legal use case by
composing the `document-vision` foundation (OCR / layout). No legal-specific
code exists in the plugin; only the interview answers differ.

## Interview transcript

```
What are you building? Describe the domain in a sentence.
> On-device legal contract review with scanned PDFs

# document-vision is detected from "scanned PDFs" / OCR language — no modality ask

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
  "notes": "must run fully offline",
  "modalities": ["document-vision"],
  "facets": ["ocr"]
}
```

## Resulting pack (excerpt)

```markdown
# MLX Research Pack: On-device legal contract review with scanned PDFs

- Roles: vision, general
- Keywords: ocr, contracts, redaction
- License filter: apache-2.0, mit
- Memory budget (GB): 32.0
- Modalities: document-vision
- Facets: ocr

## Modality foundations

- `document-vision` — Document extraction / vision
  - Facets: `ocr` (OCR / text extraction)

## Runtime preference

- Preferred: `mlx-vlm`
- Alternates: `lmstudio`
- Rationale: Vision / document / video workloads need native mlx-vlm; Ollama does not run VLMs.

## Candidates

### 1. `mlx-community/<an-ocr-vision-model>` — score 82.0/100

- Role: vision
- Suggested wiring: mlx-vlm server -> ...
- Why it ranked:
  - role_match: matched roles: vision
  - keyword_match: matched keywords: ocr
  - license_ok: license: apache-2.0
  - memory_fit: est_ram_gb ... vs budget 32.0

## Adapters / LoRAs

### 1. `org/<legal-ocr-lora>` — score 71.0/100

- Kind: adapter
- Why it ranked:
  - keyword_match: matched keywords: ocr, contracts

## Datasets

No datasets matched this intent on the Hub. See the dataset blueprint below.

## Dataset blueprint

No Hub datasets ranked for this intent. Use this guidance-only blueprint
to create a local dataset — nothing is downloaded or trained by research.

### Goal

Create a small, domain-specific dataset for **On-device legal contract review...**
```

Legal is a **composition** of the `document-vision` foundation plus domain keywords —
not a separate engine. The same foundations power audio and video packs (see sibling
examples). When Hub datasets are empty, the pack still includes an explicit dataset
blueprint so another agent can draft labeled data locally without any download or
training from research.

