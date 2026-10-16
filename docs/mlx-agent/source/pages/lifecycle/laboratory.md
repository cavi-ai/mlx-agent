# Experiments and text training

MLX Workbench is the native SwiftUI interface for the shared engine. Agent integrations can use `mlx-agent lab` with the same database, artifacts, jobs, and review records.

## Runtime

The core agent remains dependency-light. Experiments require Python 3.12 and `requirements-lab.txt`. Optional local MPS video generation uses `requirements-lab-inference.txt`. Install requirements into the selected interpreter; model weights must be present locally.

On macOS, the default experiment database and artifacts live under `~/Library/Application Support/mlx-agent/lab/`. Explicit `--database` and `--artifacts` select another workspace. Standalone `scripts/video-lab` preserves legacy JSON output.

```sh
python scripts/mlx-agent lab --help
python scripts/mlx-agent lab experiment list --json
```

Heavy generation, conversion, and training share a private compute lease. A live owner blocks conflicting launches. Receipt-backed detached jobs retain ownership; dead processes can be reclaimed, while unverifiable live identity fails closed.

## Native workflows

**Compare → Controlled studies** creates a one-variable frozen comparison, queues generation, and retains artifacts. Blind review uses a common playback timeline. Complete rubric scores and a preference are required before immutable submission and authorized reveal. Adaptive studies preserve their budgets and progression.

**Studies → New experiment → Cached model** captures recipes for the registered Wan 2.1 T2V 1.3B Diffusers/MPS adapter. `lab workspace` operations `recipe-catalog` and `recipe-prepare` inspect only the selected interpreter and local Hugging Face cache. Preparation requires an explicit cached commit, complete pipeline assets and weight shards, and installed video runtime dependencies. It hashes the weights and loader assets, records runtime and platform identity, and never downloads weights, loads a model or queues jobs. The existing preview/create operations validate and queue the controlled pair separately. Captured recipes recheck their identities before and after generation; imported legacy recipes retain their existing behavior. An MLX conversion does not replace the required Diffusers cache.

**Workspace…** previews adoption into a new destination. Stop workers and the former web application first. Adoption verifies database and artifact content, preserves IDs and sealed assignments, and retains source backups. It refuses active work, changed inputs, or an occupied destination.

**Activity** reconciles legacy conversion receipts before starting an imported pending item. Import requires a reviewed content digest and retains the source and a backup.

## Text LoRA

Training previews validate every bounded JSONL record and identify dataset content and the cached base model. Confirmation rechecks the preview and copies the dataset into a private run snapshot. Each job retains its own receipt, log, actual exit result, and verified adapter content identity.

A completed adapter can be evaluated against its pinned base model using identical prompt, seed, token limit, and sampling settings. Separate processes produce baseline and adapter outputs. The engine rechecks inputs before accepting a content-addressed report into the experiment store. Native Training exposes these retained evaluations.

An existing directory alone does not establish successful training. Historical receipts without exit evidence remain unknown; receipts without pinned base identity cannot authorize a new linked evaluation. Comparison outputs are evidence and do not automatically record a quality judgment.

Media training remains governed by its existing qualification gates. Text LoRA acceptance does not qualify video training.

## Media dataset contracts

The Python engine API in `video_lab.training_data` accepts content-addressed media, captions, source lineage, train/validation/holdout splits, and per-sample weights. It requires no character pack, outfit, reserved token, or campaign evidence. `DatasetPolicy` lets callers select minimum sample count, clip duration bounds, and a maximum weighted synthetic share; duration and synthetic-share limits are unset by default.

`freeze_dataset` rejects duplicate identities, duplicate media content, split leakage, and an empty training split. The frozen content digest binds all samples and policy settings, independent of sample order, and is checked when loading a snapshot. Caller-provided media identities and metadata must still be verified against the source bytes before training.

The persisted character-training v1 format retains its specialized validation rules. Generic dataset snapshots do not yet authorize video training jobs or provide a Workbench video-training workflow; backend qualification and job integration remain separate requirements.

The standalone metadata commands accept canonical UTF-8 JSON (sorted keys, compact separators, no trailing newline) from an explicit request root. Inputs are limited to 1 MiB; symlinks and non-regular files are rejected. They return JSON and do not open a database, read media bytes, load models, or launch training.

```sh
python scripts/mlx-agent lab training-data validate --request-root DATA --request-file dataset.json --json
python scripts/mlx-agent lab training-data freeze --request-root DATA --request-file dataset.json --json
python scripts/mlx-agent lab training-data inspect --request-root DATA --request-file frozen.json --json
```

Save the `data` object returned by `freeze` as canonical JSON for `inspect`. Frozen output orders samples by sample ID; validation failures include stable reason codes. `inspect` rechecks both readiness and the snapshot content digest.
