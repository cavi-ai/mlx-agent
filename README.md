# mlx-agent

[![PyPI version](https://img.shields.io/pypi/v/mlx-agent)](https://pypi.org/project/mlx-agent/) [![PyPI weekly downloads](https://img.shields.io/pypi/dw/mlx-agent?label=PyPI%20downloads%2Fweek)](https://pypi.org/project/mlx-agent/) [![Tests](https://img.shields.io/github/actions/workflow/status/cavi-ai/mlx-agent/test.yml?branch=main)](https://github.com/cavi-ai/mlx-agent/actions/workflows/test.yml?query=branch%3Amain)

![Your coding agent can make the props too. Local 3D generation on Apple Silicon. A duck knight holding a croissant lance.](.github/assets/mlx-agent-hero.png)

Discover, run and manage local AI models on Apple Silicon.

Generate images, video, music, speech and 3D assets with local models. Manage model discovery, conversion and serving, then keep task inputs and results together through the CLI or MCP.

## Get started

The core requires Python 3.11 or later and uses the standard library. Model execution needs a separately installed runtime; optional backends have their own dependencies.

```bash
git clone --depth 1 --branch main https://github.com/cavi-ai/mlx-agent.git
cd mlx-agent
python3 scripts/mlx-agent inspect-host --json
python3 scripts/mlx-agent discover --limit 3 --json
```

Review the candidates and follow [Your first local model](docs/mlx-agent/source/pages/introduction/quickstart.md). To prepare compatible local weights and a runtime, see [Intake](docs/mlx-agent/source/pages/lifecycle/intake.md) and [optional backends](docs/mlx-agent/source/pages/lifecycle/backends.md). Discovery does not download weights.

## Generate a 3D mesh

Turn a PNG, JPEG or WebP image into a GLB with native MLX TripoSG, TRELLIS.2 or Pixal3D. Use compatible converted local weights and the installed `mlx-3d` backend. Optional backend installation uses Python 3.12. Choose a new `.glb` path in an existing output directory:

```bash
python3 scripts/mlx-agent convert mesh --path /absolute/local/mesh-model --image /absolute/input/object.png --out /absolute/output/object.glb --json
```

For example, ask a connected agent:

> Turn this image into a GLB using my chosen local model. Show me the settings before starting, then show me the result.

For jobs that outlive the client, use [durable tasks](docs/mlx-agent/source/pages/lifecycle/tasks.md). They retain task inputs, status and output. See [local inference](docs/mlx-agent/source/pages/lifecycle/local-inference.md) for model-specific operations.

## Connect your tools

Use the CLI directly, or register the current checkout's stdio server with an MCP client:

```json
{
  "mcpServers": {
    "mlx-agent": {"command": "python3", "args": ["/absolute/path/to/mlx-agent/scripts/mlx-agent-mcp"]}
  }
}
```

See [Installation](docs/mlx-agent/source/pages/introduction/installation.md) and [provider lifecycle](docs/install/index.md) for host-specific setup, updates and removal.

## More with local models

- **Generate and understand media:** run [image, video, speech, music and 3D tasks](docs/mlx-agent/source/pages/lifecycle/local-inference.md), or use [durable tasks](docs/mlx-agent/source/pages/lifecycle/tasks.md) to retain inputs and results.
- **Classify and embed:** use local models for structured classification and text embeddings through [durable tasks](docs/mlx-agent/source/pages/lifecycle/tasks.md).
- **Find and evaluate models:** [Scout](docs/mlx-agent/source/pages/capabilities/scout.md) finds candidates; [Adopt](docs/mlx-agent/source/pages/capabilities/adopt.md) records role checks; [Bench](docs/mlx-agent/source/pages/capabilities/bench.md) measures an already-served model.
- **Connect and operate:** [Wire](docs/mlx-agent/source/pages/capabilities/wire.md) previews and applies client configuration; [Intake](docs/mlx-agent/source/pages/lifecycle/intake.md), [Convert](docs/mlx-agent/source/pages/lifecycle/convert.md), [Serve](docs/mlx-agent/source/pages/lifecycle/serve.md) and [optional backends](docs/mlx-agent/source/pages/lifecycle/backends.md) manage model and runtime lifecycle.
- **Explore and train:** use [Research](docs/mlx-agent/source/pages/capabilities/research.md), [Blueprint](docs/mlx-agent/source/pages/capabilities/blueprint.md), [Laboratory](docs/mlx-agent/source/pages/lifecycle/laboratory.md), or [LoRA and fuse](docs/mlx-agent/source/pages/lifecycle/lora-and-fuse.md).

See [Evidence and sizing](docs/mlx-agent/source/pages/reference/evidence.md) for model checks and [the security model](docs/mlx-agent/source/pages/security/model.md) for confirmation, network and recovery boundaries.

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

Local models run on [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) from Apple's ml-explore team.
