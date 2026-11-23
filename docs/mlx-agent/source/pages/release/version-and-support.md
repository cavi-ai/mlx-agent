# Version and support

Package version in this checkout: **0.16.0** (release tag `v0.16.0`).
Source documentation can include changes made after a published release. Use
the artifact attached to a specific release when you need that version's docs.

## Compatibility

| Provider | Minimum | Last tested |
| --- | --- | --- |
| Claude Code | 2.1.143 | 2.1.198 |
| Codex CLI | 0.137.0 | 0.137.0 |
| Agy | 1.1.14 | 1.1.14 |
| OpenCode | 2.3.2 (OpenCode 2 plugin API) | 2.3.2 (model discovery) |
| AgentSkills hosts | — | — |

For version checks, `doctor` compares provider releases with the minimum and
last-tested versions recorded in the compatibility metadata. A newer release
receives a `provider_untested` warning. A release below the recorded minimum is
unsupported.

Python 3.11+ for the core; macOS on Apple Silicon for host inspection and MLX
execution. Optional backend dependency installation requires Python 3.12.

## Support

- Issues and PRs: the `cavi-ai/mlx-agent` repository on GitHub.
- The core is a dependency-free Python package; adapters are generated deterministically from `plugin.json` and verified byte-for-byte in CI.

## Stable interfaces

- The eight provider capabilities (`mlx-scout`, `mlx-adopt`, `mlx-wire`, `mlx-bench`, `mlx-doctor`, `mlx-watch`, `mlx-fleet`, `mlx-task`) and their argument grammars.
- The MCP tools `mlx_agent_execute` and `mlx_agent_capabilities`.
- The confirmation-gated transaction contract (preview hash, receipts, rollback).
- Receipt and state schemas (versioned, with migrations).

CLI-only commands can gain flags in minor releases. Download and lifecycle starts
use preview hashes; direct inference and JIT policy operations use different
interfaces. Use the [CLI reference](../reference/cli.md) and command help for the
current contract.

## Release pipeline

Tagging `vX.Y.Z` runs the Release workflow. It refuses to proceed unless every committed copy of the version agrees — runtime, plugin manifest, both marketplace manifests, compatibility matrix, release evidence, site catalog, contract validator, and the documentation artifact. It then runs the full gate set, attaches `mlx-agent-docs-vX.Y.Z.tar.gz` to the GitHub Release, and dispatches `cavi-oss-release` to each registered documentation consumer.
