# OpenClaw

OpenClaw loads each installed skill as `/mlx-scout`, `/mlx-adopt`, `/mlx-wire`, `/mlx-bench`, `/mlx-doctor`, `/mlx-watch`, and `/mlx-fleet`. Every skill folder carries its own launcher and Python runtime.

Use the receipt-owned installer; it never edits `openclaw.json`:

```bash
python3 scripts/mlx-agent install openclaw --scope user --dry-run --json
python3 scripts/mlx-agent install openclaw --scope user --confirm --preview-hash <preview-hash> --json
python3 scripts/mlx-agent update openclaw --scope user --dry-run --json
python3 scripts/mlx-agent uninstall openclaw --scope user --dry-run --json
python3 scripts/mlx-agent doctor openclaw --scope user --json
```

User scope installs into the managed skills directory, `$OPENCLAW_STATE_DIR/skills` when `OPENCLAW_STATE_DIR` is set, otherwise `~/.openclaw/skills`; every agent on that state directory sees the skills. Project scope installs into `<workspace>/skills`, visible only to the agent that owns that workspace:

```bash
python3 scripts/mlx-agent install openclaw --scope project --project /absolute/agent/workspace --dry-run --json
```

Export the same `OPENCLAW_STATE_DIR` the gateway uses before running the installer. OpenClaw picks up new skills in the next session; run `openclaw skills list` and confirm the seven `mlx-*` skills appear.
