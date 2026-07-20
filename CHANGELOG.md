# Changelog

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
