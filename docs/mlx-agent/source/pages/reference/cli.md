# Command line

Run `python3 scripts/mlx-agent` from the repository root. The provider-neutral
core covers the seven host capabilities and additional model lifecycle commands.

Use `--help` at any level for current arguments, choices, and defaults:

```bash
python3 scripts/mlx-agent --help
python3 scripts/mlx-agent serve start --help
python3 scripts/mlx-agent intake fetch --help
```

## Discovery and configuration

| Command | Purpose |
| --- | --- |
| `inspect-host` | Inspect host memory and runtime inventory. |
| `discover` | Rank candidates using metadata and fit estimates. |
| `adopt start/resume/status` | Retain verification and recommendation state. |
| `research` | Write a domain research pack; `--no-write` renders only. |
| `blueprint` | Write a project design pack; `--no-write` renders only. |
| `lab` | Manage retained experiments, studies, datasets, and training through the optional engine. |
| `bench run/aggregate` | Measure an already-served model or aggregate exports. |
| `wire render/apply/status/rollback` | Render and manage configuration transactions. |
| `fleet render/apply` | Render or apply per-role LiteLLM routing configuration. |

## Model lifecycle

| Command | Purpose |
| --- | --- |
| `intake resolve` | Classify a Hugging Face source against backend registries. |
| `intake fetch/status` | Review/download a model source; inspect job receipts. |
| `intake port-analysis/port-plan` | Inspect architecture facts or draft guidance with a local model. |
| `backend list/install/remove` | Inspect or manage optional dependencies. |
| `convert scan` | Inventory GGUF files and candidate duplicates without deletion. |
| `convert start/status` | Quantize local checkpoints; inspect conversion jobs. |
| `convert transcribe/decide/generate/speak/music/describe/video` | Execute supported local inference tasks. |
| `serve start/status/stop` | Launch and inspect receipt-owned local servers. |
| `serve unload/policy` | Release JIT workers or replace an owned gateway's memory policy. |
| `lora start/status` | Train an adapter on a cached base and local dataset. |
| `fuse start/status` | Merge an adapter into standalone weights. |
| `doctor models` | Inspect model inventory, config drift, and endpoint health. |
| `watch snapshot/diff` | Retain and compare owned-model Hub state. |

## Provider lifecycle

`providers` lists detected hosts. `install`, `update`, `uninstall`, and
`doctor <provider>` manage adapters in user or project scope. Installer previews
use `--dry-run --json`; confirmed changes require the returned hash. Provider
Doctor inspects state without requiring confirmation.

## Output and exit status

Most commands support `--json` with the versioned result envelope:
`schema_version`, `operation`, `status`, `data`, `warnings`, and
`generated_at`. Errors add an `error` object with `code`, `message`,
`remediation`, and `retryable`.

Exit 0 indicates a successful command. Several mutation previews exit 2 with
`status: "ok"` and `data.requires_confirmation: true`; failures also exit 2
but have `status: "error"`. Argument-parser errors exit 2 and may not emit a
JSON envelope. Inspect the result rather than treating every exit 2 as the same
condition.

| Preview type | Hash location with `--json` |
| --- | --- |
| Provider install/update/uninstall, Wire apply, Fleet apply | `data.preview.preview_hash` |
| Serve start, Intake fetch, backend install/remove, Convert/LoRA/Fuse start, model prune | `data.plan.preview_hash` |
| Wire rollback | `data.preview.preview_hash` |

Repeat the same arguments with `--confirm --preview-hash <hash>`. Rollback
needs its own preview. Job starts return before work finishes; inspect the
relevant status and log before consuming outputs.

## State and receipts

Discovery accepts `--state-dir` or `MLX_AGENT_STATE_DIR`. Adoption uses
`--state`; Watch uses `--state-dir`. Keep paths consistent across runs.

Wire defaults to `.mlx-agent-receipts` beside its target. Serve and model jobs
use `.mlx-agent-receipts/<kind>` under the current directory, or under the base
selected by `--receipts-dir`. Provider installers have separate user/project
receipt storage. See [Security](../security/model.md).

Configuration rollback restores receipt backups. Downloads, training,
conversion, inference, server policy, and cache pruning have different lifecycle
semantics; a receipt does not make every operation reversible.
