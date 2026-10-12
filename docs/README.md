# mlx-agent documentation

Start with [Installation](mlx-agent/source/pages/introduction/installation.md), then follow
[Your first local model](mlx-agent/source/pages/introduction/quickstart.md).
These guides describe the current checkout. Published documentation comes from
the artifact attached to a release; editing source pages does not update an
already published artifact.

| Task | Guide |
| --- | --- |
| Understand the workflow and requirements | [Overview](mlx-agent/source/pages/introduction/overview.md) |
| Choose a coding-agent host and installation scope | [Installation](mlx-agent/source/pages/introduction/installation.md) · [Provider lifecycle](install/index.md) |
| Find models that fit your machine | [Scout](mlx-agent/source/pages/capabilities/scout.md) |
| Verify a model and retain the evidence | [Adopt](mlx-agent/source/pages/capabilities/adopt.md) |
| Preview, apply, or roll back a configuration | [Wire](mlx-agent/source/pages/capabilities/wire.md) |
| Measure an already-served model | [Bench](mlx-agent/source/pages/capabilities/bench.md) |
| Research a domain or design a project | [Research](mlx-agent/source/pages/capabilities/research.md) · [Blueprint](mlx-agent/source/pages/capabilities/blueprint.md) |
| Resolve and download a Hugging Face model | [Model intake](mlx-agent/source/pages/lifecycle/intake.md) |
| Manage optional converter dependencies | [Backends](mlx-agent/source/pages/lifecycle/backends.md) |
| Quantize a checkpoint or inventory GGUF files | [Convert](mlx-agent/source/pages/lifecycle/convert.md) |
| Run speech, image, video, or classification tasks | [Local inference](mlx-agent/source/pages/lifecycle/local-inference.md) |
| Serve weights and manage JIT memory | [Serve](mlx-agent/source/pages/lifecycle/serve.md) |
| Train and fuse an adapter | [LoRA and fuse](mlx-agent/source/pages/lifecycle/lora-and-fuse.md) |
| Run controlled experiments and retain training evaluations | [Laboratory](mlx-agent/source/pages/lifecycle/laboratory.md) |
| Route roles to local servers | [Fleet](mlx-agent/source/pages/lifecycle/fleet.md) |
| Inspect model health or changes | [Doctor](mlx-agent/source/pages/lifecycle/doctor.md) · [Watch](mlx-agent/source/pages/lifecycle/watch.md) |
| Diagnose a failure | [Troubleshooting](mlx-agent/source/pages/guides/troubleshooting.md) |

## References

[CLI](mlx-agent/source/pages/reference/cli.md) ·
[Runtimes](mlx-agent/source/pages/reference/runtimes.md) ·
[Evidence and sizing](mlx-agent/source/pages/reference/evidence.md) ·
[Security model](mlx-agent/source/pages/security/model.md) ·
[Transaction recovery](security.md) ·
[Version and support](mlx-agent/source/pages/release/version-and-support.md)

Domain examples: [Legal](examples/legal-research-pack.md),
[Audio](examples/audio-research-pack.md), [Video](examples/video-research-pack.md).
Runtime reference packs: [Quantization](../src/mlx_agent/resources/references/quantization.md),
[Model families](../src/mlx_agent/resources/references/model-families.md),
[Serving troubleshooting](../src/mlx_agent/resources/references/troubleshooting.md).

## Development and integration

[Contributing](../CONTRIBUTING.md) · [Development](mlx-agent/source/pages/guides/development.md) ·
[Adding a provider](adding-a-provider.md) · [Documentation consumer contract](mlx-agent/CONSUMER.md) ·
[Migrating from v0.1](migrating-from-v0.1.md)
