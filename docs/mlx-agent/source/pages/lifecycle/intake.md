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
It accepts Hugging Face repositories and local checkpoint or source directories.
Diffusers repositories are inspected through `model_index.json` and their component
configs, so a root Transformers `config.json` is not required. Local Python files
are parsed as text and never imported. Source-only directories report unknown
architecture and weight support; analysis is bounded to 4,000 files and 512 KiB
of Python source. `--revision` applies only to Hugging Face sources.

```bash
python3 scripts/mlx-agent intake port-analysis VAST-AI/TripoSG --json
python3 scripts/mlx-agent intake port-analysis /path/to/TripoSG --json
```

`port-plan` sends architecture context to an already-running local model and
writes a Markdown proposal. Review this generated guidance before implementing
a port; it does not add a backend or establish architecture support.

## Pinned model updates

For a managed update, use an immutable 40-character commit and a new absolute
`--local-dir`. A new destination must not exist; an indexed completed source can be reused in place. Repeat `--additional-file` to
include exact GGUF companions or shards beside the primary `--file`:

```bash
python3 scripts/mlx-agent intake fetch org/model --revision "$REVISION" \
  --file model.gguf --additional-file mmproj.gguf \
  --local-dir /path/to/new-source --receipts-dir /path/to/state --json
```

Inspect the preview, then repeat its arguments with `--confirm --preview-hash`.
The file set, revision and destination are bound to that hash. Additional files
require a pinned revision and local destination; shared-cache downloads are not
used for this mode. Existing `--file` requests remain supported. The same pinned file-set interface accepts configuration, tokenizer and weight files for declared HF dependencies, including safetensors and binary assets.

`intake status` includes `additional_files`, `preview_hash` and completed `files`
entries containing relative `path`, `bytes` and `signature`. These signatures
bind local metadata and bounded file samples; they are not full-file checksums.
Status rechecks managed files before reporting `done`; changed, missing or linked
payloads report `failed` with no completed path or files. The returned `path`
is the download container; append the selected `file` or `subfolder` when needed.
A completed managed fetch receipt can be passed to conversion with
`--fetch-receipt`. Changed files or a marker belonging to another launch are
refused. Managed fetch receipts and markers have immutable per-launch paths;
use the actual receipt returned by start/status, preserving older queued sources.

A pinned managed recipe fetch downloads its primary repository's files only.
Fetch each required base or dependency separately at an explicit immutable
revision, then pass its completed receipt through
`--dependency-fetch-receipt`. Recipe bases must match the declared base commit;
a mutable dependency reference is not relabeled as a resolved commit.

`required_dependencies` lists each declared HF support repository, its known
immutable revision or null, and its required files. A null revision needs an
explicit resolved commit; an empty file list denotes a snapshot dependency.
The managed primary fetch never downloads those repositories implicitly.
Manifest-declared HTTP assets are downloaded beneath `<local_dir>/support`,
checked against their declared size and SHA-256, and included in completed file
identities. Plan, receipt and status expose `downloads` and `downloads_dir`.
Pass that support directory through conversion or use the matching completed
fetch receipt; existing shared download/cache locations are not substituted.

## Reuse managed sources

Immutable fetches with `--local-dir` use a shared source index under
`$XDG_STATE_HOME/mlx-agent/model-storage` (default `~/.local/state`).
`--storage-root` selects another owned index. Preview is read-only.

The index coordinates processes across receipt directories. A verified completed
source is reused at its original location; preview reports `storage.mode: reuse`
and the effective `local_dir`. Confirmation returns `status: reused`, the original
`receipt`, and a freshly validated `reused_fetch` status record. No worker or
second destination is created. Keep the original receipt and marker for conversion.

Source identity includes the repository, immutable revision and selected file or
checkpoint. Different revisions remain separate. Overlapping snapshots, files or
companion selections are refused rather than copied again. Confirmation rechecks
admission and current file evidence. Equal, parent and child destinations cannot
be claimed by different sources. Download approval hashes bind source, recipe
and actions independently of admission state, so a reviewed failed download can
retry after Quarantine without silently approving another source. Running, changed or invalid sources block a
second download. If a stopped source has been removed or moved to quarantine, a
new preview can download it again at its original location.

A snapshot already present in the configured Hugging Face cache blocks a new
managed copy. Use that cache in place, or inspect incomplete cache files before
retrying. The index does not search arbitrary folders, migrate provider caches,
or delete model files.

`serve discover` advertises this interface as `model_storage_admission`. Managed
local destinations require immutable commits; `--hf-cache` selects cache storage
instead of copying it into a second local destination.
