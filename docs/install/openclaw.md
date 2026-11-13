# OpenClaw

OpenClaw loads each installed skill as `/mlx-scout`, `/mlx-adopt`, `/mlx-wire`, `/mlx-bench`, `/mlx-doctor`, `/mlx-watch`, `/mlx-fleet`, and `/mlx-task`. Every skill folder carries its own launcher and Python runtime.

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

Export the same `OPENCLAW_STATE_DIR` the gateway uses before running the installer. OpenClaw picks up new skills in the next session; run `openclaw skills list` and confirm the eight `mlx-*` skills appear.

## Model provider

`providers/openclaw` is an OpenClaw provider plugin named `mlx-agent`. It lists the chat models served behind a [front door](../mlx-agent/source/pages/lifecycle/serve.md#front-door) and keeps typed endpoints, which advertise `task_kinds`, out of the chat catalog.

Start the routes and the front door with one API key, then install the plugin from a checkout with OpenClaw's installer:

```bash
openclaw plugins install --link --accept-capabilities /absolute/path/to/mlx-agent/providers/openclaw
```

Point the provider at the front door and pass the key through `MLX_AGENT_API_KEY`:

```json5
{
  models: {
    providers: {
      "mlx-agent": {
        baseUrl: "http://127.0.0.1:8090/v1",
        api: "openai-completions",
        apiKey: "${MLX_AGENT_API_KEY}",
        models: [],
      },
    },
  },
}
```

`openclaw models list --refresh --provider mlx-agent` lists each chat route as `mlx-agent/<name>`. Without a `baseUrl`, the plugin uses `http://127.0.0.1:8090/v1`.

Memory search uses an embedding route through the same provider entry:

```json5
{
  memory: {
    search: { provider: "mlx-agent", model: "<embedding route name>" },
  },
}
```
