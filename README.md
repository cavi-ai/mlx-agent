# mlx-agent

[![PyPI version](https://img.shields.io/pypi/v/mlx-agent)](https://pypi.org/project/mlx-agent/) [![PyPI downloads](https://img.shields.io/pypi/dm/mlx-agent?label=PyPI%20downloads)](https://pypi.org/project/mlx-agent/) [![Tests](https://img.shields.io/github/actions/workflow/status/cavi-ai/mlx-agent/test.yml?branch=main)](https://github.com/cavi-ai/mlx-agent/actions/workflows/test.yml?query=branch%3Amain)

![Your coding agent can make the props too. Local 3D generation on Apple Silicon. A duck knight holding a croissant lance.](.github/assets/mlx-agent-hero.png)

Turn an image into a mesh saved as a GLB on your Apple Silicon Mac. With mlx-agent, your coding agent can run the same task workflow across native MLX TripoSG, TRELLIS.2 and Pixal3D models, without a separate agent integration for each.

## Start with Codex

Install the Codex CLI, then add mlx-agent from the current main branch:

```bash
codex plugin marketplace add cavi-ai/mlx-agent --ref main
codex plugin add mlx-agent@mlx-agent
```

Restart Codex. Before generating, choose an image and a compatible local model, and check that its backend is installed. The [backend guide](docs/mlx-agent/source/pages/lifecycle/backends.md) explains how to inspect and install `mlx-3d`; [Intake](docs/mlx-agent/source/pages/lifecycle/intake.md) and [Convert](docs/mlx-agent/source/pages/lifecycle/convert.md) cover obtaining and preparing weights. Review the previews and confirm any setup you choose. Optional backends have separate dependencies; `mlx-3d` uses Python 3.12.

Then ask Codex:

> Use mlx-agent and my chosen compatible local model to turn this image into a GLB. Show me the settings for approval, then check the task and show me the resulting file.

Codex can use `$mlx-agent:mlx-task` to run, track and collect the task. The job retains its input and output, so you can return to the result and inspect the GLB in your preferred 3D tool. See [durable tasks](docs/mlx-agent/source/pages/lifecycle/tasks.md) and [local inference](docs/mlx-agent/source/pages/lifecycle/local-inference.md).

<details>
<summary>Install for other hosts</summary>

Install your coding-agent host first and restart it after adding mlx-agent. See [Installation](docs/mlx-agent/source/pages/introduction/installation.md) and [provider lifecycle](docs/install/index.md) for scope, updates, removal and recovery.

For checkout-based routes, clone current main:

```bash
git clone --depth 1 --branch main https://github.com/cavi-ai/mlx-agent.git
```

### Claude Code

```bash
claude plugin marketplace add cavi-ai/mlx-agent
claude plugin install mlx-agent@mlx-agent
```

### Agy

From a checkout of current main:

```bash
agy plugin validate ./mlx-agent/providers/agy
agy plugin install ./mlx-agent/providers/agy
```

### OpenCode, OpenClaw, Hermes Agent and AgentSkills-compatible hosts

From the checkout, select `opencode`, `openclaw`, `hermes` or `agentskills` as the provider. For example, for OpenCode:

```bash
cd mlx-agent
python3 scripts/mlx-agent install opencode --scope user --dry-run --json
# Review the plan and copy data.preview.preview_hash.
python3 scripts/mlx-agent install opencode --scope user --confirm --preview-hash <preview-hash> --json
python3 scripts/mlx-agent doctor opencode --scope user --json
```

For portable AgentSkills, the installer copies a runtime into each installed skill. Move unowned older `mlx-*` skill folders aside before the first managed install. OpenClaw and Hermes use their respective state directories. See the [provider guides](docs/install/index.md) for host-specific details.

### Other MCP hosts

From a checkout of current main, register the stdio server with your MCP host:

```json
{
  "mcpServers": {
    "mlx-agent": {"command": "python3", "args": ["/absolute/path/to/mlx-agent/scripts/mlx-agent-mcp"]}
  }
}
```

Provider installation copies plugin files and runs from a checkout.

</details>

## More with local models

- **Generate and understand media:** run [image, video, speech, music and 3D tasks](docs/mlx-agent/source/pages/lifecycle/local-inference.md), or use [durable tasks](docs/mlx-agent/source/pages/lifecycle/tasks.md) to retain inputs and results.
- **Classify and embed:** use local models for structured classification and text embeddings through [durable tasks](docs/mlx-agent/source/pages/lifecycle/tasks.md).
- **Find and evaluate models:** [Scout](docs/mlx-agent/source/pages/capabilities/scout.md) finds candidates; [Adopt](docs/mlx-agent/source/pages/capabilities/adopt.md) records role checks; [Bench](docs/mlx-agent/source/pages/capabilities/bench.md) measures an already-served model.
- **Connect and operate:** [Wire](docs/mlx-agent/source/pages/capabilities/wire.md) previews and applies client configuration; [Intake](docs/mlx-agent/source/pages/lifecycle/intake.md), [Convert](docs/mlx-agent/source/pages/lifecycle/convert.md), [Serve](docs/mlx-agent/source/pages/lifecycle/serve.md) and [optional backends](docs/mlx-agent/source/pages/lifecycle/backends.md) manage model and runtime lifecycle.
- **Explore and train:** use [Research](docs/mlx-agent/source/pages/capabilities/research.md), [Blueprint](docs/mlx-agent/source/pages/capabilities/blueprint.md), [Laboratory](docs/mlx-agent/source/pages/lifecycle/laboratory.md), or [LoRA and fuse](docs/mlx-agent/source/pages/lifecycle/lora-and-fuse.md).

The Python core uses the standard library. Model execution needs a separately installed runtime; optional backends have their own dependencies. See [Evidence and sizing](docs/mlx-agent/source/pages/reference/evidence.md) for model checks and [the security model](docs/mlx-agent/source/pages/security/model.md) for confirmation, network and recovery boundaries.

<details>
<summary>Full host and skill reference</summary>

<!-- compatibility:begin -->
## Provider support

First-class adapters are included for each provider below. The universal installer supports both user and project scopes.

| Provider | Package | Invoke |
| --- | --- | --- |
| [Claude Code](docs/install/claude.md) | Native plugin | `/mlx-scout`<br>`/mlx-adopt`<br>`/mlx-wire`<br>`/mlx-bench`<br>`/mlx-doctor`<br>`/mlx-watch`<br>`/mlx-fleet`<br>`/mlx-task` |
| [Codex CLI](docs/install/codex.md) | Native plugin | `$mlx-agent:mlx-scout`<br>`$mlx-agent:mlx-adopt`<br>`$mlx-agent:mlx-wire`<br>`$mlx-agent:mlx-bench`<br>`$mlx-agent:mlx-doctor`<br>`$mlx-agent:mlx-watch`<br>`$mlx-agent:mlx-fleet`<br>`$mlx-agent:mlx-task` |
| [Agy](docs/install/agy.md) | Native plugin | `mlx-scout skill`<br>`mlx-adopt skill`<br>`mlx-wire skill`<br>`mlx-bench skill`<br>`mlx-doctor skill`<br>`mlx-watch skill`<br>`mlx-fleet skill`<br>`mlx-task skill` |
| [OpenCode](docs/install/opencode.md) | Native plugin | `/mlx-scout`<br>`/mlx-adopt`<br>`/mlx-wire`<br>`/mlx-bench`<br>`/mlx-doctor`<br>`/mlx-watch`<br>`/mlx-fleet`<br>`/mlx-task` |
| [OpenClaw](docs/install/openclaw.md) | Native skills | `/mlx-scout`<br>`/mlx-adopt`<br>`/mlx-wire`<br>`/mlx-bench`<br>`/mlx-doctor`<br>`/mlx-watch`<br>`/mlx-fleet`<br>`/mlx-task` |
| [Hermes Agent](docs/install/hermes.md) | Native skills | `/mlx-scout`<br>`/mlx-adopt`<br>`/mlx-wire`<br>`/mlx-bench`<br>`/mlx-doctor`<br>`/mlx-watch`<br>`/mlx-fleet`<br>`/mlx-task` |
| [AgentSkills-compatible hosts](docs/install/index.md) | Portable skills | `mlx-scout skill`<br>`mlx-adopt skill`<br>`mlx-wire skill`<br>`mlx-bench skill`<br>`mlx-doctor skill`<br>`mlx-watch skill`<br>`mlx-fleet skill`<br>`mlx-task skill` |
<!-- compatibility:end -->

</details>

## Documentation and contributing

Start with the [documentation index](docs/README.md), [CLI reference](docs/mlx-agent/source/pages/reference/cli.md), [runtimes](docs/mlx-agent/source/pages/reference/runtimes.md), or [troubleshooting](docs/mlx-agent/source/pages/guides/troubleshooting.md). See [Contributing](CONTRIBUTING.md) for local checks and [Development](docs/mlx-agent/source/pages/guides/development.md) for adapters and documentation builds.

Versioned documentation comes from `docs/mlx-agent/source`; the [consumer contract](docs/mlx-agent/CONSUMER.md) describes release artifacts. Distributed under the [MIT License](LICENSE).
