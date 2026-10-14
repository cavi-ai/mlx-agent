# Troubleshooting

Run commands from the same checkout and with the same state and receipt roots
used for the original operation. Start with:

```bash
python3 scripts/mlx-agent inspect-host --json
python3 scripts/mlx-agent doctor models --wired-root . --json
python3 scripts/mlx-agent serve status --json
```

## A preview exits with status 2

Several commands return 2 for a successful preview that still needs confirmation.
With `--json`, check `status`, `data.requires_confirmation`, and the preview hash
before treating it as a failure. A result with `status: "error"` carries an error
code and remediation. See [CLI output](../reference/cli.md#output-and-exit-status).

## A preview hash is stale

Generate a new preview and review it. Keep the command's model, path, runtime,
port, and policy arguments identical when adding `--confirm --preview-hash`.
Changing those arguments or the files they describe can invalidate the hash.
Apply and rollback use separate previews and separate hashes.

## Skills or commands are missing

Restart the host after installing. Run
`python3 scripts/mlx-agent doctor <provider> --scope user --json` from the checkout.
Use the same project scope if that is where you installed. Codex invokes skills
as `$mlx-agent:mlx-scout`; Agy uses named skills. For portable AgentSkills, use
the universal installer so the runtime is copied with each skill. Unowned older
skill folders must be moved aside before the installer can manage them.

## Discovery fails offline

`offline_cache_missing` means there is no matching cache entry for that request
policy. Run online once with the same filters and `--state-dir`, then retry
offline. `--offline` applies to discovery metadata; adoption can still inspect
already-running local runtimes.

## A candidate is not verified or a benchmark cannot run

`runtime_inventory` or metadata evidence does not mean generation succeeded.
Make the selected model available in its runtime, then start a new adoption run.
`bench` requires an already-served model and its exact runtime model identifier.
Use [Serve](../lifecycle/serve.md) for local MLX weights; Ollama and LM Studio
manage their own loading. Tool-use checks do not execute real tools, and
`mlx-vlm` reports `unsupported-runtime` for that probe.

## An optional backend is missing

Use `python3 scripts/mlx-agent backend list --json` to see installed state and
log paths. Optional backend installation requires Python 3.12; the dependency-free
core's Python 3.9 minimum does not apply to those locks. Review and confirm the
[backend installation](../lifecycle/backends.md) instead of installing dependencies
into an unrelated Python interpreter.

## A server died or a wired configuration changed

Doctor reports missing models, changed/missing wired files, port conflicts, and
endpoint health. Serve status cross-checks its receipts against processes.
Inspect the receipt's log path for startup failure; preserve receipts and backups
if recovery is required. Do not stop an unrelated process to make a port free.
For a changed configuration, review a fresh apply or [rollback](../capabilities/wire.md#recover-a-configuration).

## JIT requests return HTTP 503

`memory_unknown` means headroom or local weight size could not be established;
`insufficient_headroom` means the admission estimate exceeds available memory.
Inspect `memory_check` and `load_blocked_reason` in serve status. Close other
memory consumers or use smaller weights, then retry. The gateway remains
reachable. See [JIT memory policy](../lifecycle/serve.md#memory-policy).

If loading or generation fails after admission, reduce context and concurrent
memory use. Discovery fit estimates and JIT admission checks do not reserve
memory against other applications or later KV-cache growth.
