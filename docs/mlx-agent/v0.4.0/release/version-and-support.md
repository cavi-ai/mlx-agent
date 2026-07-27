# Version and support

Current release: **0.4.0** (tag `v0.4.0`).

## Compatibility

| Provider | Minimum | Last tested |
| --- | --- | --- |
| Claude Code | 2.1.143 | 2.1.198 |
| Codex CLI | 0.137.0 | 0.137.0 |
| Gemini CLI | 0.46.0 | 0.46.0 |
| OpenCode | 1.17.7 | 1.18.3 |
| AgentSkills hosts | — | — |

Python 3.9+; macOS on Apple Silicon for host inspection and MLX execution.

## Support

- Issues and PRs: the `cavi-ai/mlx-agent` repository on GitHub.
- The core is a dependency-free Python package; adapters are generated deterministically from `plugin.json` and verified byte-for-byte in CI.

## What is stable in 0.4.x

- The four provider capabilities (`mlx-scout`, `mlx-adopt`, `mlx-wire`, `mlx-bench`) and their argument grammars.
- The confirmation-gated transaction contract (preview hash, receipts, rollback).
- Receipt and state schemas (versioned, with migrations).

CLI-only lifecycle commands (`doctor`, `serve`, `convert`, `lora`, `fuse`, `fleet`, `watch`) follow the same contract but may gain flags in minor releases.
