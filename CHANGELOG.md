# Changelog

## Unreleased

- Enrich research packs with ranked PEFT/LoRA adapters and Hub datasets (hybrid list + card scoring via the existing scorer), emit a deterministic dataset blueprint when no datasets match, and write a JSON sidecar beside the markdown pack. Still read-only: no downloads.
- Document verified tool-use recommendations and safety boundaries, and add an opt-in Apple Silicon smoke test that probes the first installed candidate on supported loopback runtimes.
- Add `mlx-agent research`: read-only domain research packs. An interview builds a validated domain intent; a transparent scoring core ranks models from Hugging Face metadata and bounded model-card text; results are written as project-local markdown under `mlx-research/`. No verification, wiring, or downloads.

## 0.3.0 - 2026-07-20

- Route OpenCode user-scope artifacts through `XDG_CONFIG_HOME` while preserving native `HOME`.
- Route user-scope installer receipts through `XDG_STATE_HOME` unless `MLX_AGENT_CONFIG_ROOT` is explicitly set.
- Record OpenCode 1.18.3 native command discovery and the isolated install/uninstall lifecycle.
- Document complete provider invocation, installation, update, verification, and recovery paths.
- Retain confirmation-gated, receipt-owned mutations and provider-specific command syntax.

## 0.2.0 - 2026-07-17

- Added the provider-neutral Scout, Adopt, and Wire core.
- Added native Claude Code, Codex CLI, Gemini CLI, and OpenCode adapters plus portable AgentSkills packages.
- Added deterministic generation, compatibility contracts, transactional installation, and recovery evidence.

## 0.1.0

- Initial Claude marketplace release and legacy Scout workflow.
