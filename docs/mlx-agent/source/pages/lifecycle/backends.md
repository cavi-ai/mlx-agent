# Optional backends

Backend manifests describe converters and model types. Built-in `mlx-lm` uses
the main runtime environment. Optional backends have separate locked environments
for vision, audio, embeddings/classification, image generation, and video generation.

## Inspect availability

```bash
python3 scripts/mlx-agent backend list --json
```

The result reports each backend's ID, installed state, categories, registry
source, environment target, and job log. Use the returned IDs rather than
assuming a model is supported by a package name.

## Install

Use Python 3.12 for optional backend installation: the shipped dependency locks
target that interpreter, even though the core supports Python 3.9+.

```bash
python3 scripts/mlx-agent backend install mlx-audio --json
python3 scripts/mlx-agent backend install mlx-audio --confirm --preview-hash <hash> --json
python3 scripts/mlx-agent backend list --json
```

Review the target, lock digest, dependency count, network hosts, and command.
Confirmation starts an installation job using the shipped lock. Check installed
state and its log before converting or running inference; a returned PID alone
does not establish successful installation.

These installs download dependencies from PyPI. They do not install the coding-agent
host or fetch model weights. Existing targets are refused; removal is a separate
reviewed action. Built-in backends are not managed by this installer.

## Remove

```bash
python3 scripts/mlx-agent backend remove mlx-audio --json
python3 scripts/mlx-agent backend remove mlx-audio --confirm --preview-hash <hash> --json
```

Removal moves the selected environment to the Trash. Review the source and
destination, and stop using that environment before removal. Reinstallation
requires a new preview and confirmation.

See [Intake](intake.md), [Convert](convert.md), and [Local inference](local-inference.md)
for resolving, quantizing, and executing models.
