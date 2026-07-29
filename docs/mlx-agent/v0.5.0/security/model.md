# Security model

mlx-agent is built around one rule: nothing irreversible or external happens without a reviewed preview and an explicit confirmation.

## Never

- **Never downloads models.** Discovery, verification, bench, serve, convert, lora, and fuse all require the model to be local first.
- **Never installs anything.** Provider CLIs and model runtimes are detected, not installed.
- **Never persists secrets.** Configs reference `MLX_AGENT_LOCAL_API_KEY`; resolved credentials in existing files are rejected before any write.
- **Never touches remote endpoints.** Runtime probes are loopback-only: no redirects, no credentials in URLs, no query strings, absolute deadlines, bounded responses.

## Always

- **Preview-then-confirm.** Every mutation (wire, fleet, serve, convert, lora, fuse, launchd, prune) renders the exact plan first. `--confirm --preview-hash` must match the reviewed preview or the operation fails stale.
- **Receipts.** Applied transactions, server processes, and batch jobs all leave receipts: hashes, pids, argv, logs. Rollback restores bytes exactly; stop signals only receipt-verified processes.
- **Deterministic probes.** Role verification uses fixed synthetic prompts with exact validators — never LLM-judged, never user data.
- **Bounded everything.** Request sizes, response sizes, line counts, iteration counts, and concurrency all have hard caps.

## Verification, not metadata

A model is recommended for a role only on evidence: runtime generation, deterministic role probes, or measured performance. Download counts and tags inform ranking but never count as verification. Tool-use specifically requires a schema-valid synthetic runtime tool call.

## Prune is the exception

`doctor models --prune` deletes files irreversibly — by design, and it says so loudly. It only targets incomplete cache snapshots, names every directory in the preview, and re-verifies cache containment at delete time.
