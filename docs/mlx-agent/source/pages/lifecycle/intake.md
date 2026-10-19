# Model intake

Intake classifies a Hugging Face source and identifies an available MLX backend.
Resolution and architecture analysis read metadata. Fetching weights is a
separate confirmed operation.

## Resolve a source

```bash
python3 scripts/mlx-agent intake resolve <repo-or-hugging-face-url> --json
```

Review the verdict, model type, task, backend installation state, components,
and warnings. A registry match is conversion guidance, not evidence that the
model has run. Sources can identify a repository, revision, subfolder, or file.
Use `--revision` to select a revision explicitly.

The `mlx_conversions` field lists up to eight MLX models that the Hub records as
quantized from the source repository, ordered by downloads. Each entry includes
`repo`, `downloads`, and `library_name`. The lookup is omitted for sources already
classified as MLX and for subfolder sources. A failed lookup leaves the list
empty and adds a warning without changing the resolution verdict.

These are metadata matches, not verified runtime results. Review each candidate's
revision, license, size, and backend compatibility before fetching it.

`cached` is true when the local Hugging Face cache already holds every file
`intake fetch` would download for the resolved revision; a fetch then downloads
nothing. Ported model types and recipe sources always report false.

## Review and fetch

```bash
python3 scripts/mlx-agent intake fetch <repo> --revision <revision> --json
python3 scripts/mlx-agent intake fetch <repo> --revision <revision> --confirm --preview-hash <hash> --json
python3 scripts/mlx-agent intake status --json
```

The preview identifies the repository, revision, file patterns, destination,
and executable arguments. Fetch requires `huggingface_hub` in the selected
environment. It starts a receipt-tracked job; inspect status and the log before
using its output. Review license, gating, storage, and source revision before
downloading.

Use `--file <relative-file>` for one GGUF file, `--hf-cache <cache-root>` for
a different cache, or `--local-dir <absolute-directory>` for an explicit
destination. If resolve supplies a ported `model_type`, pass it with
`--model-type` to use the declared file selection. Default snapshot fetching
excludes formats MLX does not consume, including GGUF; choose a GGUF file explicitly.

Keep arguments identical between preview and confirmation. `--receipts-dir`
selects the base for fetch receipts; use that same base for status. A returned
PID indicates a started job, not a finished download.

## Convert or serve

Already-usable MLX weights can be [served](serve.md). Original checkpoints use
[Convert](convert.md); optional backends may need an explicit
[installation](backends.md). GGUF conversion quality is limited by its source.

## Analyze an unsupported architecture

```bash
python3 scripts/mlx-agent intake port-analysis <repo> --json
python3 scripts/mlx-agent intake port-plan <repo> --endpoint http://127.0.0.1:8080/v1 --model <served-model-id> --json
```

`port-analysis` reports deterministic metadata facts without downloading weights.
`port-plan` sends architecture context to an already-running local model and
writes a Markdown proposal. Review this generated guidance before implementing
a port; it does not add a backend or establish architecture support.
