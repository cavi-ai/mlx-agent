# Scout

Scout discovers MLX models on Hugging Face, bucketed by role for this host: `general`, `coding`, `reasoning`, `vision`, `embedding`, plus `tool-use` membership, and the task roles `speech-to-text`, `text-to-speech`, `image-generation`, `video-generation`, `mesh-generation` and `music-generation`.

```bash
python3 scripts/mlx-agent discover --role coding --limit 3
python3 scripts/mlx-agent discover --role text-to-speech --limit 3
python3 scripts/mlx-agent discover --new          # sort by Hub lastModified
python3 scripts/mlx-agent discover --sort downloads --limit 4
python3 scripts/mlx-agent discover --sort downloads --search "qwen3 coder"
python3 scripts/mlx-agent discover --fast         # skip enrichment
python3 scripts/mlx-agent discover --exclude-local --json
```

## Roles

`general`, `coding`, `reasoning`, `vision` and `embedding` come from the
repository name. A task role lists the Hub by its pipeline tag:

| Role | Pipeline tag |
| --- | --- |
| `speech-to-text` | `automatic-speech-recognition` |
| `text-to-speech` | `text-to-speech` |
| `image-generation` | `text-to-image` |
| `video-generation` | `text-to-video` |
| `mesh-generation` | `image-to-3d` |
| `music-generation` | `text-to-audio`, with a music term in the name or tags |

Without `--role`, a model's pipeline tag places it in its task role before any
name keyword applies; embedding tags place it in `embedding`. Models whose tag
names a task no role covers, such as classification, are not listed. Each
candidate's `facts.pipeline_tag` records the tag, and its role provenance says
whether the role came from the pipeline tag or the name. Task roles point to
`mlx-agent task capabilities` for backend compatibility, and `--runtime`
refuses them for chat runtimes.

## Local models

`--local-status` reads the local model folders once (`convert scan` without
content signatures). The Markdown report shows each candidate's local status,
and the JSON report adds a `local` object to every candidate:

| Field | Meaning |
| --- | --- |
| `status` | `exact`: the same repository is local, or a same-named model whose weight files total the same bytes. `variant`: another quantization or format of the same model name is local. `absent`: neither. |
| `basis` | `repository`, `weight_bytes`, `model_key` or `null`. |
| `model_key` | The repository name without format and quantization labels, as `convert scan` keys local models. |
| `copies` | Each matching local model: `path`, `repo`, `store`, `format`, `quantization`, `acquired`. |

`--exclude-local` implies `--local-status` and leaves out every model with an
`exact` or `variant` copy before ranking, so every returned slot is a model
you do not have. The response's `local` object reports `models` (local models
read), `exclude_local` and `excluded` (`repo`, `status`, `basis`, `copies`).
Vision projectors and speculative drafters never count as copies.

Local folders default to the well-known model directories, including LM
Studio's configured downloads folder; `--gguf-root`, `--mlx-root` and
`--receipts-dir` select them as in `convert scan`. Cached responses are keyed
by the local models they were built against.

## Lenses and search

`--sort` ranks each role by a Hugging Face order instead of the publisher-first
ranking: `downloads`, `trending`, `likes`, or `newest` (creation date, limited to
the reputable publishers). Each candidate reports `list_rank`, its position in
that listing, so a client can merge roles into one ranked list. `--search` sends
its text to the Hub's repository-name search.

Repositories tagged `gguf` without `safetensors` are left out: they hold no MLX
weights. Discovery inspects repositories in rank order and stops once every
requested role's list is final, so a small `--limit` sends few Hub requests.

## What makes the ranking honest

- **Real sizing** — actual quantized byte size from the HF tree API, not a name guess.
- **Context-aware fit** — architecture (layers, KV heads, head dim) from HF config gives each candidate an `estimates.kv` block with the maximum context your RAM supports. `--context N` tightens `fits` to weights + KV at that context.
- **Reasoning detection** — reads the chat template and tags; reasoning models stay out of fast/cheap roles.
- **Quant dedup** — `-4bit / -8bit / -bf16` variants roll up to one logical model and the best quant that fits wins.
- **License and gating** — surfaced before any runtime fetch.
- **Community bench** — when the bundled aggregate has matching numbers for your chip, candidates carry a `community_bench` annotation.

## Filters

`--role`, `--limit`, `--memory-gb`, `--quantization`, `--license`, `--publisher`, `--runtime`, `--exclude-gated`, `--exclude-local`, `--offline`, `--refresh`, and `--context` compose freely. `discover --wire <repo> --target <runtime>` emits a ready config block for a chosen model.

`--offline` and `--refresh` are mutually exclusive. Offline discovery requires
a cache created by an earlier online request with matching policy. Use a
consistent `--state-dir`, or supply `MLX_AGENT_STATE_DIR`:

```bash
python3 scripts/mlx-agent discover --role coding --state-dir .mlx-agent/scout-cache --json
python3 scripts/mlx-agent discover --role coding --state-dir .mlx-agent/scout-cache --offline --json
```

Missing cache returns `offline_cache_missing`. Fast mode skips enrichment;
inspect each field's evidence before treating a value as a fetched fact.

## Tool-use and next steps

A model retains its primary role while also carrying `tool-use` membership.
Metadata is not verification. A `metadata-only` status can accompany either
`metadata_only` or `heuristic_only` strength; inspect both fields.

Use [Adopt](adopt.md) after making the selected model available locally.
[Evidence and sizing](../reference/evidence.md) explains fit and probe limits.
