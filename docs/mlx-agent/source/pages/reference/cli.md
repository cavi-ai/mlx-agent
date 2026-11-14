# Command line

Run `python3 scripts/mlx-agent` from the repository root. The provider-neutral
core covers the eight host capabilities and additional model lifecycle commands.
`python3 scripts/mlx-agent --version` prints the installed version.

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
| `bench run/aggregate/coding` | Measure an already-served model or aggregate exports. |
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
| `convert transcribe/decide/generate/speak/music/describe/video/mesh` | Execute supported local inference tasks. |
| `task capabilities/start/status/list/result/cancel/verify-model` | Manage durable local classification and media tasks. See [Durable inference tasks](../lifecycle/tasks.md). |
| `serve start/status/stop` | Launch and inspect receipt-owned local servers. |
| `serve discover` | Find services across registered receipt locations and installed runtime inventories. |
| `serve protected-service/private-worker` | Run authenticated supervised services or private workers for an existing service manager. |
| `serve unload/policy` | Release JIT workers or update an owned gateway's memory policy. |
| `serve profile save/show/list/delete` | Save and reuse serving memory, headroom, residency and execution settings. |
| `serve workflow save/show/list/delete/preview/start/status/stop/resolve` | Define, review and manage a named set of local serving endpoints. |
| `lora inspect/start/status/stop` | Inspect prerequisites, train an adapter, inspect jobs, or stop owned training. |
| `fuse start/status` | Merge an adapter into standalone weights. |
| `doctor models` | Inspect model inventory, config drift, and endpoint health. |
| `watch snapshot/diff` | Retain and compare owned-model Hub state. |

### Serving workflows

A serving workflow binds named roles to existing local models, runtime settings and ports. Create the resource profiles first with `serve profile save`; the workflow refers to those profiles by name. For example, save a JSON file such as:

```json
{"schema":"serving-workflow/1","name":"studio","members":{
  "coding":{"path":"/models/coder","runtime":"mlx_lm","port":8086,"profile":"coding"},
  "classifier":{"path":"/models/classifier","runtime":"tasks","port":8085,"profile":"small"}
}}
```

Use local paths or already cached repositories supported by the selected runtime. Serving does not download models. Each member may specify an `api_key_file` reference; keep the key itself out of the workflow file.

```sh
mlx-agent serve workflow save --name studio --file workflow.json
mlx-agent serve workflow preview --name studio --json
mlx-agent serve workflow start --name studio --confirm --preview-hash PREVIEW_HASH
mlx-agent serve workflow status --name studio --json
mlx-agent serve workflow resolve --name studio --role classifier --json
mlx-agent serve workflow stop --name studio
```

Copy `preview_hash` from the reviewed preview. The resource forecast is advisory: an unknown estimate is not a fit guarantee, and admission checks current memory when work starts. A reachable endpoint can be unloaded; JIT and task models load on their first request and follow the chosen residency policy afterward. If activation reports a partial state, inspect workflow status and use workflow stop to clean up its created endpoints. Reused endpoints are left running.

Resolution returns `serving-workflow-role/1` with the selected endpoint's `origin`, `base_url`, `model_id`, `runtime`, supported `operations`, `task_kinds`, optional `credential_ref`, and process identity. The origin has no `/v1` suffix; the base URL includes it once. For typed tasks, send exactly `{"kind": ..., "parameters": ...}` to that role's `/v1/tasks` endpoint. Use the same endpoint for that job's status, result, artifacts and cancellation. Select roles explicitly even when models support the same task kind.

`convert video` and `convert mesh` accept `--denoiser-cache adaptive` and
`--cache-memory-mib` for supported Wan2.1 T2V and TripoSG models. These caches
are off by default and may change frames or geometry. The same settings are
available in typed task requests. See [Local inference](../lifecycle/local-inference.md).

## Provider lifecycle

| Command | Purpose |
| --- | --- |
| `providers` | Inspect supported coding-agent hosts. |
| `install/update/uninstall` | Preview and manage owned provider files and receipts. |
| `doctor <provider>` | Check provider installation, ownership and native visibility. |

See [Installation](../introduction/installation.md) for host-specific commands
and user/project scopes.

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
| Serve start, Intake fetch, backend install/remove, Convert/LoRA/Fuse start, `doctor models --prune` | `data.plan.preview_hash` |
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
