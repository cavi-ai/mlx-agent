# Version and support

Current release: **0.5.0** (tag `v0.5.0`).

## Compatibility

| Provider | Minimum | Last tested |
| --- | --- | --- |
| Claude Code | 2.1.143 | 2.1.198 |
| Codex CLI | 0.137.0 | 0.137.0 |
| Gemini CLI | 0.46.0 | 0.46.0 |
| OpenCode | 1.17.7 | 1.18.9 |
| AgentSkills hosts | — | — |

Python 3.9+; macOS on Apple Silicon for host inspection and MLX execution.

## Support

- Issues and PRs: the `cavi-ai/mlx-agent` repository on GitHub.
- The core is a dependency-free Python package; adapters are generated deterministically from `plugin.json` and verified byte-for-byte in CI.

## What is stable in 0.5.x

- The seven provider capabilities (`mlx-scout`, `mlx-adopt`, `mlx-wire`, `mlx-bench`, `mlx-doctor`, `mlx-watch`, `mlx-fleet`) and their argument grammars.
- The confirmation-gated transaction contract (preview hash, receipts, rollback).
- Receipt and state schemas (versioned, with migrations).

CLI-only lifecycle commands (`serve`, `convert`, `lora`, `fuse`) follow the same contract but may gain flags in minor releases.

## Release pipeline

Tagging `vX.Y.Z` runs the Release workflow. It refuses to proceed unless every committed copy of the version agrees — runtime, plugin manifest, both marketplace manifests, compatibility matrix, release evidence, site catalog, contract validator, and the documentation artifact. It then runs the full gate set, attaches `mlx-agent-docs-vX.Y.Z.tar.gz` to the GitHub Release, and dispatches `cavi-oss-release` to each registered documentation consumer.
