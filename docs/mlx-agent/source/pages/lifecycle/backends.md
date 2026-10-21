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

These installs download dependencies from PyPI. The audio backend also installs
the hash-pinned English spaCy pipeline from its official GitHub release; those
hosts and the build-tool lock digest appear in its preview. It includes Misaki's
English phonemizer and checks that it produces phonemes before marking the
installation successful. English Kokoro generation therefore needs no runtime
language-pipeline download. Other Kokoro languages can require additional dependencies.
Kokoro uses a private temporary alias for eSpeak data so deeply nested backend
paths remain usable. The alias is removed when generation finishes.
When a local Kokoro directory contains the requested voice under `voices/`,
speech synthesis uses that safetensors file directly, including the default
`af_heart` voice, without requiring a separate Hugging Face cache entry.

The audio lock includes the source-only `docopt` dependency of `num2words`.
Only that package may build from its hash-pinned source, using separately locked
build tools without build isolation. All other dependencies remain binary-only.

Backend installation does not install the coding-agent host or fetch speech-model
weights. Existing targets are refused; removal is a separate reviewed action.
An older audio environment missing Misaki needs removal and reinstallation to
receive these dependencies; model directories are separate and are not removed.
Built-in backends are not managed by this installer.

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
