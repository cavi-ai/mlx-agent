# Command line

The `mlx-agent` CLI is the provider-neutral core every adapter wraps. Standard library only; JSON envelopes for agents, human output for people.

## Discovery and verification

| Command | Purpose |
| --- | --- |
| `discover` | Discover MLX models for this host. |
| `inspect-host` | Host and runtime inventory without discovery. |
| `adopt start/resume/status` | Resumable verify-and-recommend workflow. |
| `bench run/aggregate` | Measure a served model; aggregate export files. |

## Configuration and operation

| Command | Purpose |
| --- | --- |
| `wire render/apply/status/rollback` | Confirmation-gated config transactions. |
| `serve start/stop/status` | Confirmation-gated server lifecycle; `--launchd` for agents. |
| `fleet render/apply` | One-shot per-role router config. |
| `doctor models` | Inventory, drift, endpoint health; `--prune` for broken snapshots. |
| `watch snapshot/diff` | Stateful owned-model Hub digest. |

## Production

| Command | Purpose |
| --- | --- |
| `convert scan` | Read-only GGUF inventory: converted, pending, and redundant copies. |
| `convert start/status` | Confirmation-gated quantization (4/8bit) from a cached repo or a GGUF file. |
| `lora start/status` | Confirmation-gated LoRA training. |
| `fuse start/status` | Confirmation-gated adapter fusion. |

## Research and design

| Command | Purpose |
| --- | --- |
| `research` | Read-only domain research packs (markdown + JSON). |
| `blueprint` | Guidance-only project design packs. |

## Provider lifecycle

`providers`, `install`, `update`, `uninstall`, and `doctor` manage the provider adapters themselves — all preview-then-confirm, all receipt-owned.

Every mutating command follows the same contract: preview without `--confirm` exits 2 with a diff and hash; `--confirm --preview-hash` applies exactly the reviewed plan; receipts make every mutation inspectable and reversible (except prune, which is loudly irreversible).
