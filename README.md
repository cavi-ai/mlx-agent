# mlx-agent

Discover, verify, and operate local MLX models on Apple Silicon for your coding
agent. mlx-agent ranks candidates for your memory budget, records local role
checks, measures served models, and writes reviewed connection configurations.

The Python core uses the standard library. Model execution needs a separately
installed runtime; optional backends have their own dependencies.

Current package version: **0.8.0**. These docs describe the current checkout;
published releases may contain earlier behavior.

## Install

Choose your coding-agent host. Install its CLI first, then restart it after
adding mlx-agent. See [Installation](docs/mlx-agent/source/pages/introduction/installation.md)
for requirements and [provider lifecycle](docs/install/index.md) for project
scope, updates, removal, and recovery.

### Claude Code

```bash
claude plugin marketplace add cavi-ai/mlx-agent
claude plugin install mlx-agent@mlx-agent
```

Run `/mlx-scout` after restarting.

### Codex CLI

```bash
codex plugin marketplace add cavi-ai/mlx-agent --ref v0.8.0
codex plugin add mlx-agent@mlx-agent
```

Invoke `$mlx-agent:mlx-scout` after restarting. Codex does not support custom
slash commands.

### Agy

```bash
git clone --depth 1 --branch v0.8.0 https://github.com/cavi-ai/mlx-agent.git
agy plugin validate ./mlx-agent/providers/agy
agy plugin install ./mlx-agent/providers/agy
```

Restart Agy, check `/skills`, and ask it to use `mlx-scout`.

### OpenCode and portable skills

From a release checkout:

```bash
git clone --depth 1 --branch v0.8.0 https://github.com/cavi-ai/mlx-agent.git
cd mlx-agent
python3 scripts/mlx-agent install opencode --scope user --dry-run --json
# Review the plan; copy data.preview.preview_hash.
python3 scripts/mlx-agent install opencode --scope user --confirm --preview-hash <preview-hash> --json
python3 scripts/mlx-agent doctor opencode --scope user --json
```

Restart OpenCode and use `/mlx-scout`. For AgentSkills-compatible hosts, use
the same preview/confirm sequence with `agentskills`:

```bash
python3 scripts/mlx-agent install agentskills --scope user --dry-run --json
python3 scripts/mlx-agent install agentskills --scope user --confirm --preview-hash <preview-hash> --json
```

The installer copies a runtime into each installed portable skill. Move unowned
older `mlx-*` skill folders aside before the first managed install.

### OpenClaw and Hermes Agent

From the same checkout, use the preview/confirm sequence with `openclaw` or
`hermes`:

```bash
python3 scripts/mlx-agent install openclaw --scope user --dry-run --json
python3 scripts/mlx-agent install openclaw --scope user --confirm --preview-hash <preview-hash> --json
```

Skills go to `$OPENCLAW_STATE_DIR/skills` (default `~/.openclaw/skills`) or
`$HERMES_HOME/skills` (default `~/.hermes/skills`). Start a new session and use
`/mlx-scout`.

### Other MCP hosts

The `mlx-agent` package on PyPI carries the CLI and the MCP server. Add the
server to any MCP host's configuration:

```json
{
  "mcpServers": {
    "mlx-agent": {"command": "uvx", "args": ["--from", "mlx-agent", "mlx-agent-mcp"]}
  }
}
```

Install the CLI with `uv tool install mlx-agent` or `pipx install mlx-agent`.
Provider installation copies the plugin files, so `mlx-agent install` runs from
a checkout.

## First local model

From a checkout, inspect the host and find candidates:

```bash
python3 scripts/mlx-agent inspect-host --json
python3 scripts/mlx-agent discover --role coding --limit 3 --json
```

Review size, license, gating, runtime compatibility, and evidence. Discovery
does not download weights. Make the chosen model available with your runtime
or an explicitly confirmed [Intake](docs/mlx-agent/source/pages/lifecycle/intake.md)
download, then start its local server.

Intake can also list existing MLX conversions of a source model on the Hub.
Review these candidates before downloading or converting weights.

Follow [Your first local model](docs/mlx-agent/source/pages/introduction/quickstart.md)
to load weights, verify the shortlist, measure performance, and preview a
configuration. Keep the returned receipts for inspection and rollback.

## What you can do

| Goal | Guide |
| --- | --- |
| Find models that fit a role and context budget | [Scout](docs/mlx-agent/source/pages/capabilities/scout.md) |
| Test candidates and retain evidence | [Adopt](docs/mlx-agent/source/pages/capabilities/adopt.md) |
| Apply or recover a connection configuration | [Wire](docs/mlx-agent/source/pages/capabilities/wire.md) |
| Measure an already-served model | [Bench](docs/mlx-agent/source/pages/capabilities/bench.md) |
| Research a domain or design a project | [Research](docs/mlx-agent/source/pages/capabilities/research.md) · [Blueprint](docs/mlx-agent/source/pages/capabilities/blueprint.md) |
| Download, quantize, and run model tasks | [Intake](docs/mlx-agent/source/pages/lifecycle/intake.md) · [Convert](docs/mlx-agent/source/pages/lifecycle/convert.md) · [Local inference](docs/mlx-agent/source/pages/lifecycle/local-inference.md) |
| Serve models with JIT loading and memory policy | [Serve](docs/mlx-agent/source/pages/lifecycle/serve.md) |
| Run controlled experiments and retain training evaluations | [Laboratory](docs/mlx-agent/source/pages/lifecycle/laboratory.md) |
| Train or fuse an adapter | [LoRA and fuse](docs/mlx-agent/source/pages/lifecycle/lora-and-fuse.md) |
| Route roles and inspect model health or changes | [Fleet](docs/mlx-agent/source/pages/lifecycle/fleet.md) · [Doctor](docs/mlx-agent/source/pages/lifecycle/doctor.md) · [Watch](docs/mlx-agent/source/pages/lifecycle/watch.md) |

## Understand the evidence

Models retain a primary role and may also have `tool-use` membership. For
tool-use, metadata is not verification: recommendation requires a
schema-valid synthetic runtime tool call. The bounded probe supports Ollama and local
OpenAI-compatible LM Studio, `mlx_lm`, and LiteLLM servers.

Fit estimates and synthetic role probes help choose what to evaluate; they do
not establish production quality or reserve memory. See
[Evidence and sizing](docs/mlx-agent/source/pages/reference/evidence.md).

Configuration transactions and model job starts use reviewed preview hashes.
Explicit `intake fetch` downloads weights, and `backend install` installs
optional dependencies. Direct inference has a different execution contract.
Read the [security model](docs/mlx-agent/source/pages/security/model.md)
for confirmation, network, ownership, and recovery boundaries.

<!-- compatibility:begin -->
## Provider support

First-class adapters are included for each provider below. The universal installer supports both user and project scopes.

| Provider | Package | Invoke |
| --- | --- | --- |
| [Claude Code](docs/install/claude.md) | Native plugin | `/mlx-scout`<br>`/mlx-adopt`<br>`/mlx-wire`<br>`/mlx-bench`<br>`/mlx-doctor`<br>`/mlx-watch`<br>`/mlx-fleet` |
| [Codex CLI](docs/install/codex.md) | Native plugin | `$mlx-agent:mlx-scout`<br>`$mlx-agent:mlx-adopt`<br>`$mlx-agent:mlx-wire`<br>`$mlx-agent:mlx-bench`<br>`$mlx-agent:mlx-doctor`<br>`$mlx-agent:mlx-watch`<br>`$mlx-agent:mlx-fleet` |
| [Agy](docs/install/agy.md) | Native plugin | `mlx-scout skill`<br>`mlx-adopt skill`<br>`mlx-wire skill`<br>`mlx-bench skill`<br>`mlx-doctor skill`<br>`mlx-watch skill`<br>`mlx-fleet skill` |
| [OpenCode](docs/install/opencode.md) | Native plugin | `/mlx-scout`<br>`/mlx-adopt`<br>`/mlx-wire`<br>`/mlx-bench`<br>`/mlx-doctor`<br>`/mlx-watch`<br>`/mlx-fleet` |
| [OpenClaw](docs/install/openclaw.md) | Native skills | `/mlx-scout`<br>`/mlx-adopt`<br>`/mlx-wire`<br>`/mlx-bench`<br>`/mlx-doctor`<br>`/mlx-watch`<br>`/mlx-fleet` |
| [Hermes Agent](docs/install/hermes.md) | Native skills | `/mlx-scout`<br>`/mlx-adopt`<br>`/mlx-wire`<br>`/mlx-bench`<br>`/mlx-doctor`<br>`/mlx-watch`<br>`/mlx-fleet` |
| [AgentSkills-compatible hosts](docs/install/index.md) | Portable skills | `mlx-scout skill`<br>`mlx-adopt skill`<br>`mlx-wire skill`<br>`mlx-bench skill`<br>`mlx-doctor skill`<br>`mlx-watch skill`<br>`mlx-fleet skill` |
<!-- compatibility:end -->

## Documentation and contributing

The [documentation index](docs/README.md) links onboarding, workflows,
troubleshooting, references, and examples. Start with
[CLI](docs/mlx-agent/source/pages/reference/cli.md),
[Runtimes](docs/mlx-agent/source/pages/reference/runtimes.md), or
[Troubleshooting](docs/mlx-agent/source/pages/guides/troubleshooting.md).

See [Contributing](CONTRIBUTING.md) for local checks and
[Development](docs/mlx-agent/source/pages/guides/development.md) for adapters and
documentation builds. The opt-in release live smoke selects only direct local
Ollama, LM Studio, or `mlx_lm` backends.
LiteLLM is excluded from automatic live selection because it can route to remote or paid backends.

Versioned documentation comes from `docs/mlx-agent/source`. The
[consumer contract](docs/mlx-agent/CONSUMER.md) describes release artifacts;
editing source pages does not refresh an already published release.

Distributed under the [MIT License](LICENSE).
