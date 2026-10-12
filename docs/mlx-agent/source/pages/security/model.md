# Security model

mlx-agent separates metadata inspection, local execution, confirmed configuration
changes, and explicit downloads. Confirmation and recovery rules differ by command.

## Network access and installation

Discovery, Research, Watch, and intake analysis read Hub metadata. Discovery
can use a matching local cache with `--offline`. Adoption and Bench do not
download weights or start servers.

`intake fetch` explicitly downloads a reviewed model source.
`backend install` explicitly installs dependencies from shipped locks.
Both require confirmation. Provider CLIs remain separately installed. Serve,
Convert, LoRA, and Fuse require local model inputs and do not fetch missing weights.

Runtime verification and health checks restrict endpoints to loopback with
bounded requests. LiteLLM can still route remotely: route locality matters,
not only the listener address.

## Confirmation and recovery

Wire/Fleet apply, Wire rollback, provider install/update/uninstall, lifecycle
starts, backend removal, and model prune require a reviewed preview hash.
Changed arguments or state require a new preview.

Wire and provider transactions keep non-secret receipts and backups.
Configuration rollback restores recorded bytes after checking current state.
Server stop targets receipt-owned processes. Job receipts record process and
output state; they do not offer general rollback for downloads, training,
generated weights, or dependency installation.

Direct inference, research/design pack writes, Watch snapshots, and JIT
unload/policy commands have their own validated interfaces; they are not all
preview-confirmed. Prune irreversibly deletes selected incomplete snapshots.

## Credentials and local writers

Rendered connection configs reference `MLX_AGENT_LOCAL_API_KEY`; Wire refuses
resolved secrets in configuration writes. Receipts are not credential stores.
Inspect inputs, job logs, and receipt paths before sharing them.

Wire uses advisory locks for cooperative writers. A process that ignores them
can race a filesystem mutation. Use permissions and process isolation when
hostile local writers are in scope. Installer ownership checks refuse unowned
or user-modified artifacts rather than overwriting them.

## Synthetic verification

Role probes use fixed bounded prompts and deterministic validators. Tool-use
requires a schema-valid synthetic runtime tool call; metadata is not verification.
The probe supplies a synthetic definition and never executes a real user tool.
Normalized evidence is retained instead of raw probe prompts, responses,
endpoints, or credentials. This does not describe logs from separate user-driven
inference commands.

See [Evidence and sizing](../reference/evidence.md) for probe limits,
[Troubleshooting](../guides/troubleshooting.md) for failures, and
[Wire](../capabilities/wire.md) for configuration recovery.
