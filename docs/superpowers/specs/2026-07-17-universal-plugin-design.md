# Universal MLX Agent Plugin Design

**Status:** Approved design

**Date:** 2026-07-17

**Target release:** Universal provider support following v0.1.0

## Summary

`mlx-agent` will become a universal, provider-native plugin for Claude Code, Codex, Gemini CLI, and OpenCode while retaining a generic AgentSkills distribution. Each first-class provider will expose the same native commands: `/mlx-scout`, `/mlx-adopt`, and `/mlx-wire`.

The implementation will keep the dependency-free Python runtime as the execution core, introduce a canonical capability manifest, generate or validate provider-native adapters from that contract, and add a universal installer alongside each provider's native installation path. The installer will support both user-global and project-local scopes.

First-class support requires manifest and schema validation, isolated install/update/uninstall round-trip tests, and real provider CLI smoke tests in CI wherever the provider can be automated. Apple Silicon end-to-end checks remain necessary because generic Linux CI cannot prove host inspection or local MLX runtime behavior.

## Goals

- Provide native installation and native slash commands for Claude Code, Codex, Gemini CLI, and OpenCode.
- Preserve capability parity for Scout, Adopt, and Wire across all first-class providers.
- Retain generic AgentSkills compatibility for other compatible agents.
- Provide a universal installer that detects providers and supports user and project scopes.
- Make configuration writes transactional, previewable, reversible, and validated.
- Replace prompt-parsed workflow state with stable, versioned structured contracts.
- Improve model discovery, verification, recommendation, wiring, diagnostics, and cross-session continuity.
- Prevent provider adapters and documentation from drifting apart.

## Non-goals

- Installing provider CLIs automatically.
- Downloading models without explicit user confirmation.
- Hiding provider-specific capabilities behind a lowest-common-denominator interface.
- Adding a persistent service, MCP server, or required package-manager runtime.
- Supporting every agent client as a first-class target in the initial universal release.
- Storing tokens, credentials, or secret values in generated configuration, receipts, logs, or evidence bundles.

## Current State

The v0.1.0 repository has four portability layers:

1. A dependency-free Python discovery and wiring core.
2. A self-contained AgentSkills-compatible `mlx-scout` skill.
3. Claude-specific commands, agent instructions, and Workflow orchestration.
4. Claude-specific marketplace and plugin metadata.

The core is largely portable, but the command layer assumes `${CLAUDE_PLUGIN_ROOT}`, and `/mlx-adopt` depends on Claude's Workflow primitive. Packaging alone cannot provide equivalent behavior on other providers. Adoption orchestration must move into a provider-neutral core workflow while native provider adapters retain the best UX each platform supports.

## Architecture

The repository will contain six bounded layers:

```text
Provider-neutral product contract
        |
        +-- Core Python CLI: inspect, discover, verify, recommend, wire
        +-- Canonical capability definitions: Scout, Adopt, Wire
        +-- Provider adapters
        |     +-- Claude Code
        |     +-- Codex
        |     +-- Gemini CLI
        |     +-- OpenCode
        +-- Generic AgentSkills distribution
        +-- Universal installer: install, update, uninstall, doctor
        +-- Validation suite: schemas, round trips, CLI smoke tests
```

### Core Python CLI

The Python core owns deterministic operations and remains compatible with Python 3.9+ using the standard library. It performs host inspection, Hugging Face discovery, model enrichment and ranking, installed-runtime probing, candidate verification, recommendation generation, configuration rendering, safe application, health checks, and rollback.

Provider prompts orchestrate this core. They do not reimplement ranking, configuration, or transaction logic.

### Canonical plugin manifest

A versioned canonical manifest is the source of truth for:

- Plugin identity, version, supported platforms, and executable requirements.
- Capabilities, arguments, aliases, and user-visible descriptions.
- Shared prompt fragments and mandatory safety constraints.
- Provider feature mappings and provider-specific extensions.
- Installation artifacts and destinations for each supported scope.
- Permissions, expected side effects, and confirmation requirements.
- Minimum supported provider versions and compatibility status.
- Validation and smoke-test commands.

Every provider artifact declares which canonical capability it implements. Generated-artifact and parity checks fail when a capability is missing or stale.

Provider-specific files are allowed when a platform offers unique functionality. They remain thin adapters and must not duplicate deterministic core behavior.

### Provider adapters

Each first-class provider exposes `/mlx-scout`, `/mlx-adopt`, and `/mlx-wire` as native slash commands. Natural-language skill activation is an additional path, not a substitute for native commands.

| Provider | Native package surface | Commands | Adoption execution |
| --- | --- | --- | --- |
| Claude Code | Marketplace plugin, commands, skills, agent, workflow adapter | `/mlx-scout`, `/mlx-adopt`, `/mlx-wire` | Native orchestration adapter over the core workflow |
| Codex | Codex-native plugin, skills, commands, and agent instructions | `/mlx-scout`, `/mlx-adopt`, `/mlx-wire` | Native delegation when available, sequential fallback |
| Gemini CLI | Extension manifest, commands, skills, and context | `/mlx-scout`, `/mlx-adopt`, `/mlx-wire` | Gemini-native orchestration over the core workflow |
| OpenCode | Commands, agents, skills, and optional plugin/config adapter | `/mlx-scout`, `/mlx-adopt`, `/mlx-wire` | Native subtasks when available, sequential fallback |
| Other AgentSkills clients | Self-contained skill | Natural-language activation | Sequential core workflow |

Native formats and minimum versions must be based on each provider's public contract at implementation time. The design does not infer one provider's schema from another provider's format.

Gemini CLI extensions support installation from GitHub or a local path and can bundle commands, skills, context, hooks, and MCP configuration. OpenCode supports native commands, agents, skills, local plugins, and npm plugins. These public contracts should inform the respective adapters rather than introducing custom loaders.

References:

- [Gemini CLI extensions](https://geminicli.com/docs/extensions/)
- [Gemini CLI extension reference](https://geminicli.com/docs/extensions/reference/)
- [OpenCode plugins](https://opencode.ai/docs/plugins/)
- [OpenCode commands](https://opencode.ai/docs/commands/)
- [OpenCode agents](https://opencode.ai/docs/agents/)

## Capability Contracts

### Scout

Scout discovers and explains suitable MLX models for the current host. It will add:

- Installed-model and running-runtime inventory.
- Filters for role, memory budget, quantization, license, gated status, publisher, and runtime.
- Ranking explanations for selected and rejected candidates.
- Cached discovery with visible freshness and explicit refresh.
- Offline use of the last successful discovery result.
- A stable, schema-validated JSON response.
- Explicit labels separating measured facts, repository metadata, estimates, and heuristics.

### Adopt

Adopt becomes a provider-neutral, resumable state machine:

```text
inspect -> discover -> shortlist -> verify -> compare -> recommend
```

The core records workflow state and evidence so an interrupted run can resume across sessions or providers. Verification concurrency is bounded according to host resources. A provider may use native subagents to perform independent verification work, but the same state contract also supports sequential execution.

Adopt never downloads a missing model automatically. It distinguishes locally runtime-tested candidates from candidates assessed only through metadata.

Its result includes:

- One recommended model per requested role.
- Alternatives and explicit rejection reasons.
- Memory and runtime constraints.
- Verification method and evidence strength.
- An exact wiring preview.
- A portable evidence bundle for later inspection or handoff.

### Wire

Wire becomes a configuration transaction:

```text
detect -> render -> diff -> confirm -> back up -> apply -> parse -> health-check
```

Runtime and provider configuration formats are isolated behind adapters. Every successful mutation produces a receipt containing non-secret file paths, hashes, adapter versions, backup references, and validation results. The receipt supports status inspection and precise rollback.

Secrets are represented only as environment-variable references. Wire must not print, copy, persist, or back up resolved secret values.

## Structured Tool Surface

The core exposes stable structured operations:

```text
inspect-host
discover
inspect-model
verify
recommend
render-config
apply-config
doctor
```

The exact CLI grouping may be refined during implementation, but each operation must have a versioned request and result schema. Results include:

- `schema_version`
- operation status
- structured data
- warnings and errors with stable codes
- metadata provenance
- timestamps and freshness
- machine-readable remediation and suggested next actions

Human-readable output remains the default for terminal users. Provider agents request JSON and consume bounded result summaries rather than parsing Markdown tables.

## Universal Installer

The dependency-free installer is exposed through the Python CLI:

```text
mlx-agent providers
mlx-agent install [provider...] --scope user|project
mlx-agent update [provider...]
mlx-agent uninstall [provider...]
mlx-agent doctor [provider...]
```

If providers are omitted, the installer detects supported CLIs and presents the targets it found. Users may select one or multiple providers.

Every mutating operation uses the same transaction protocol:

1. Detect the provider and its version.
2. Resolve the requested user or project destination.
3. Validate compatibility and required executables.
4. Produce a dry-run plan showing files and configuration changes.
5. Request explicit confirmation.
6. Back up files that will change and create a pending receipt.
7. Stage changes and install atomically.
8. Validate manifests, configuration syntax, and provider discovery.
9. Run a minimal capability smoke test.
10. Commit the receipt on success or restore the previous state on failure.

Provider-native installation remains documented and supported. The universal installer is optional and does not replace native package managers or marketplaces.

The installer must not:

- Install a provider CLI.
- Download a model.
- Overwrite unrelated provider configuration.
- Change configuration without a preview and confirmation.
- Persist credentials or secret values.
- Claim support when the provider version is outside the validated range.

## State, Receipts, and Handoffs

User-scope state lives in an OS-appropriate application data directory. Project-scope state lives under a dedicated ignored directory in the project. State paths must not collide with provider-owned configuration directories.

Three distinct artifacts prevent ambiguous recovery:

- **Installation receipt:** what adapter artifacts were installed and where.
- **Configuration transaction receipt:** what Wire changed and how to restore it.
- **Adoption handoff:** resumable workflow state, evidence, and remaining steps.

All artifacts are schema-versioned, exclude secrets, and support migration or a clear incompatibility error.

## Error Handling

Errors use stable categories:

- provider not installed or unsupported version
- manifest or schema incompatibility
- missing executable or runtime
- network unavailable or rate limited
- authentication or gated-model access required
- insufficient memory or storage
- corrupt provider configuration
- interrupted or conflicting transaction
- runtime load or generation failure
- validation or health-check failure

Every failure identifies whether the operation changed state. Failed mutations roll back automatically when safe. If automatic rollback cannot finish, the receipt records the exact recovery command and affected paths.

Agent-facing results distinguish retryable failures from user-action and compatibility failures. Retries are bounded and never conceal a required confirmation, model download, authentication step, or configuration conflict.

## Agent Ergonomics

Provider prompts remain small workflow guides around structured core operations. They share:

- Consistent command names and arguments.
- Bounded output defaults to preserve context.
- Explicit next-action hints.
- Stable failure categories and remediation.
- A capability matrix describing the installed adapter and provider features.
- A standard handoff artifact for continuing Adopt across sessions or providers.
- The same confirmation gates for downloads and configuration writes.

Provider-native orchestration is an optimization. It must not change recommendation semantics or safety requirements.

## Compatibility Matrix

A committed compatibility matrix records, for every provider:

- Adapter version.
- Minimum and last tested CLI versions.
- Supported install scopes.
- Native command, skill, agent, and orchestration features.
- Configuration files the adapter may modify.
- Required and optional executables.
- Scout, Adopt, and Wire parity status.
- Last successful real smoke-test environment and date.

Documentation and release tooling generate user-visible support claims from this matrix. Unsupported or untested combinations are labeled accurately.

## Validation Strategy

### Core and schema tests

- Unit tests for discovery, ranking, verification, rendering, receipts, health checks, and rollback.
- JSON Schema tests for the canonical manifest and all structured operations.
- Schema migration and incompatible-version tests.
- Golden-file tests for generated provider artifacts.
- Checks that generated files match the canonical source.

### Installer tests

- Install, update, and uninstall round trips in isolated temporary user homes and projects.
- Multiple-provider and mixed-scope installations.
- Idempotent reinstall and no-op update behavior.
- Corrupt configuration, interrupted writes, stale backups, and conflicting receipts.
- Missing runtime, offline cache, unsupported provider version, and permission failures.
- Verification that unrelated configuration and files are preserved.

### Provider tests

- Manifest and schema validation for every provider package.
- Provider discovery tests proving all three native commands are recognized.
- Real CLI smoke tests in CI wherever licensing, authentication, and automation permit.
- Scheduled or manually triggered compatibility jobs where hosted CI cannot legally or technically run a provider.
- Test results written back to the compatibility matrix through a reviewed release workflow.

### Apple Silicon tests

Apple Silicon testing proves host inspection, local runtime detection, candidate verification, and safe wiring against supported runtimes. Linux CI may validate portable discovery and schemas but cannot satisfy the product's end-to-end release gate by itself.

## Release Gates

A universal release is ready only when:

- Generated provider artifacts match the canonical contract.
- Every first-class provider exposes native Scout, Adopt, and Wire commands.
- Capability parity checks pass.
- Installer and configuration rollback round trips pass.
- Provider discovery smoke tests pass for supported versions.
- Apple Silicon end-to-end checks pass.
- Fixtures, logs, receipts, and evidence bundles contain no secret-like values.
- Native and universal install documentation matches tested commands.
- The compatibility matrix reflects current proof.
- A clean installation reaches a successful `doctor` result for every supported provider.

## Documentation

Documentation will be organized around user goals:

- A quick-start provider selector with native and universal installation paths.
- Provider-specific installation, scope, update, removal, and troubleshooting guides.
- Scout, Adopt, and Wire task guides.
- Security, confirmation, and side-effect model.
- Current compatibility matrix.
- Schema and provider-adapter extension guide.
- Contributor workflow for adding another provider.
- Migration guide for existing Claude users that preserves current command names.

## Implementation Order

The universal release is one product milestone implemented in dependency order:

1. Define canonical schemas and structured core operations.
2. Move Adopt orchestration into the resumable core workflow.
3. Build the installer transaction, receipt, rollback, and doctor foundations.
4. Adapt the existing Claude package to the canonical contract.
5. Add Codex, Gemini CLI, and OpenCode native adapters.
6. Add generic AgentSkills packaging and compatibility checks.
7. Generate installation documentation and the compatibility matrix.
8. Add isolated round-trip, provider CLI, and Apple Silicon validation.

Implementation planning must split these dependencies into reviewable slices and keep generated artifacts separate from handwritten provider extensions.

## Acceptance Criteria

- A fresh user can install the plugin natively on any first-class provider.
- A fresh user can use the universal installer for one or multiple detected providers at user or project scope.
- All four providers recognize `/mlx-scout`, `/mlx-adopt`, and `/mlx-wire`.
- Equivalent inputs produce schema-equivalent results across providers.
- Adopt can resume after interruption and distinguishes runtime-tested from metadata-only evidence.
- Wire previews every change, requires confirmation, validates the result, and can roll back precisely.
- Provider adapters are generated or validated from one canonical contract.
- CI proves schemas, installation round trips, and available real CLIs; Apple Silicon validation proves the MLX-specific path.
- Documentation never claims support beyond the committed compatibility evidence.
