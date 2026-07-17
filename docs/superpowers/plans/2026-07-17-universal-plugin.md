# Universal MLX Agent Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `mlx-agent` into a dependency-free universal plugin with native Scout, Adopt, and Wire surfaces for Claude Code, Codex, Gemini CLI, and OpenCode, plus generic AgentSkills packaging and a safe multi-provider installer.

**Architecture:** Extract the current script into a focused `src/mlx_agent` standard-library package, define one versioned capability manifest and structured result contract, and generate provider-native adapters from that source. A transaction-based installer applies user- or project-scoped artifacts, while the core Adopt and Wire state machines provide provider-independent behavior with native orchestration as an optional optimization.

**Tech Stack:** Python 3.9+ standard library, JSON/JSON Schema documents, Markdown provider artifacts, Node.js only for the existing Claude Workflow adapter, `unittest`, provider CLIs in opt-in smoke jobs, GitHub Actions.

## Global Constraints

- Runtime compatibility remains Python 3.9+ with standard-library-only production code.
- First-class providers are Claude Code, Codex, Gemini CLI, and OpenCode.
- Claude Code, Gemini CLI, and OpenCode expose native `/mlx-scout`, `/mlx-adopt`, and `/mlx-wire` commands. Codex exposes installed `$mlx-agent:mlx-scout`, `$mlx-agent:mlx-adopt`, and `$mlx-agent:mlx-wire` skills; current Codex does not support custom slash commands.
- Generic AgentSkills compatibility remains supported.
- Scout, Adopt, and Wire have schema-equivalent results across providers.
- User-global and project-local installation scopes are both supported.
- Every configuration mutation is previewed, explicitly confirmed, backed up, validated, receipted, and rolled back on failure.
- Provider CLIs and missing models are never installed or downloaded automatically.
- Secrets are represented by environment-variable references and never persisted in logs, receipts, evidence bundles, or fixtures.
- Compatibility claims come only from the committed compatibility matrix and current validation evidence.

---

## File Structure

### Core package

- `src/mlx_agent/__init__.py` — package version and public API.
- `src/mlx_agent/__main__.py` — `python3 -m mlx_agent` entry point.
- `src/mlx_agent/cli.py` — argument parsing, human/JSON rendering, exit codes.
- `src/mlx_agent/contracts.py` — result envelopes, error codes, JSON serialization.
- `src/mlx_agent/host.py` — Apple Silicon and local-runtime inventory.
- `src/mlx_agent/huggingface.py` — Hugging Face API client and cache.
- `src/mlx_agent/models.py` — model records, classification, sizing, and provenance.
- `src/mlx_agent/discovery.py` — filtering, deduplication, ranking, and explanations.
- `src/mlx_agent/verification.py` — bounded runtime/model verification.
- `src/mlx_agent/adoption.py` — resumable Adopt state machine and evidence bundle.
- `src/mlx_agent/wiring.py` — runtime config renderers and health checks.
- `src/mlx_agent/transactions.py` — preview, backup, atomic apply, receipt, rollback.
- `src/mlx_agent/providers.py` — canonical provider registry and path resolution.
- `src/mlx_agent/installer.py` — detect/install/update/uninstall/doctor orchestration.
- `scripts/mlx-agent` — repository-local executable bootstrap.
- `skills/mlx-scout/scripts/scout.py` — backward-compatible wrapper around the package.

### Contracts and generated adapters

- `plugin.json` — canonical plugin/capability manifest.
- `schemas/plugin.schema.json` — canonical manifest schema.
- `schemas/result.schema.json` — structured operation envelope schema.
- `schemas/adoption-state.schema.json` — resumable Adopt state schema.
- `schemas/receipt.schema.json` — install and configuration receipt schema.
- `providers/claude/**` — generated Claude Code package.
- `providers/codex/**` — generated Codex package.
- `providers/gemini/**` — generated Gemini CLI extension.
- `providers/opencode/**` — generated OpenCode package.
- `providers/agentskills/**` — generated generic AgentSkills package.
- `scripts/generate_adapters.py` — deterministic adapter generator.
- `scripts/validate_contracts.py` — standard-library schema and parity validator.

### Tests and release evidence

- `tests/unit/**` — pure core tests.
- `tests/integration/**` — cache, Adopt, Wire, and installer transaction tests.
- `tests/contracts/**` — manifest, generated artifact, and parity tests.
- `tests/smoke/**` — provider CLI discovery scripts.
- `compatibility/providers.json` — tested provider/version/capability matrix.
- `.github/workflows/test.yml` — portable unit, contract, and round-trip gates.
- `.github/workflows/provider-smoke.yml` — opt-in real CLI tests.
- `.github/workflows/apple-silicon.yml` — Apple Silicon end-to-end gate.

---

### Task 1: Establish the structured core and canonical manifest

**Files:**
- Create: `src/mlx_agent/__init__.py`
- Create: `src/mlx_agent/contracts.py`
- Create: `plugin.json`
- Create: `schemas/plugin.schema.json`
- Create: `schemas/result.schema.json`
- Create: `scripts/validate_contracts.py`
- Create: `tests/contracts/test_manifest.py`
- Create: `tests/unit/test_contracts.py`

**Interfaces:**
- Produces: `ResultEnvelope.ok(operation, data, warnings=[])`, `ResultEnvelope.fail(operation, code, message, remediation, retryable=False)`, and `ResultEnvelope.to_dict()`.
- Produces: canonical capability IDs `scout`, `adopt`, and `wire`, each mapped to native command name `mlx-<id>`.
- Produces: `validate_manifest(path: Path) -> list[str]` and `validate_result(value: dict) -> list[str]`.

- [ ] **Step 1: Write failing contract tests**

```python
# tests/unit/test_contracts.py
import unittest
from mlx_agent.contracts import ResultEnvelope

class ResultEnvelopeTests(unittest.TestCase):
    def test_success_envelope_is_versioned(self):
        value = ResultEnvelope.ok("inspect-host", {"chip": "Apple M4"}).to_dict()
        self.assertEqual(value["schema_version"], "1.0")
        self.assertEqual(value["operation"], "inspect-host")
        self.assertEqual(value["status"], "ok")
        self.assertEqual(value["data"]["chip"], "Apple M4")
        self.assertEqual(value["warnings"], [])

    def test_error_envelope_exposes_remediation(self):
        value = ResultEnvelope.fail(
            "discover", "network_unavailable", "HF unavailable",
            "Retry with --offline to use the last cache.", retryable=True,
        ).to_dict()
        self.assertEqual(value["status"], "error")
        self.assertTrue(value["error"]["retryable"])
        self.assertIn("--offline", value["error"]["remediation"])
```

```python
# tests/contracts/test_manifest.py
import json, unittest
from pathlib import Path
from scripts.validate_contracts import validate_manifest

ROOT = Path(__file__).resolve().parents[2]

class ManifestTests(unittest.TestCase):
    def test_manifest_has_three_capabilities_and_four_native_providers(self):
        manifest = json.loads((ROOT / "plugin.json").read_text())
        self.assertEqual(set(manifest["capabilities"]), {"scout", "adopt", "wire"})
        self.assertEqual(set(manifest["providers"]), {"claude", "codex", "gemini", "opencode", "agentskills"})
        self.assertEqual(validate_manifest(ROOT / "plugin.json"), [])
        for provider in ("claude", "gemini", "opencode"):
            self.assertEqual(
                manifest["providers"][provider]["commands"],
                ["mlx-scout", "mlx-adopt", "mlx-wire"],
            )
        self.assertEqual(
            manifest["providers"]["codex"]["commands"],
            ["mlx-agent:mlx-scout", "mlx-agent:mlx-adopt", "mlx-agent:mlx-wire"],
        )
```

- [ ] **Step 2: Run tests and verify import/manifest failures**

Run: `PYTHONPATH=src python3 -m unittest tests.unit.test_contracts tests.contracts.test_manifest -v`

Expected: FAIL because `mlx_agent.contracts`, `plugin.json`, and `scripts.validate_contracts` do not exist.

- [ ] **Step 3: Implement the result envelope and canonical manifest**

```python
# src/mlx_agent/contracts.py
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"

@dataclass(frozen=True)
class ErrorDetail:
    code: str
    message: str
    remediation: str
    retryable: bool = False

@dataclass(frozen=True)
class ResultEnvelope:
    operation: str
    status: str
    data: Dict[str, Any] = field(default_factory=dict)
    warnings: List[Dict[str, str]] = field(default_factory=list)
    error: Optional[ErrorDetail] = None
    schema_version: str = SCHEMA_VERSION
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def ok(cls, operation: str, data: Dict[str, Any], warnings=None):
        return cls(operation=operation, status="ok", data=data, warnings=list(warnings or []))

    @classmethod
    def fail(cls, operation: str, code: str, message: str, remediation: str, retryable=False):
        return cls(operation=operation, status="error", error=ErrorDetail(code, message, remediation, retryable))

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        if self.error is None:
            value.pop("error")
        return value
```

Create `plugin.json` with identity `mlx-agent`, schema version `1.0`, the three capability definitions, the five provider entries, supported scopes `user` and `project`, and production requirement `python3 >=3.9`. Define command arguments once under each capability and reference them from providers.

Create Draft 2020-12 JSON schemas that require the exact top-level keys asserted above. Implement `scripts/validate_contracts.py` as a focused standard-library validator for required keys, types, enum values, command parity, and schema-version equality; do not add a production JSON Schema dependency.

- [ ] **Step 4: Run focused tests**

Run: `PYTHONPATH=src python3 -m unittest tests.unit.test_contracts tests.contracts.test_manifest -v`

Expected: all focused tests pass and the validator prints no errors.

- [ ] **Step 5: Commit the structured contract**

```bash
git add src/mlx_agent plugin.json schemas scripts/validate_contracts.py tests/unit/test_contracts.py tests/contracts/test_manifest.py
git commit -m "feat: define universal plugin contracts"
```

### Task 2: Extract the current Scout implementation into focused modules

**Files:**
- Create: `src/mlx_agent/host.py`
- Create: `src/mlx_agent/huggingface.py`
- Create: `src/mlx_agent/models.py`
- Create: `src/mlx_agent/discovery.py`
- Create: `src/mlx_agent/cli.py`
- Create: `src/mlx_agent/__main__.py`
- Create: `scripts/mlx-agent`
- Modify: `skills/mlx-scout/scripts/scout.py`
- Test: `tests/unit/test_models.py`
- Test: `tests/integration/test_scout_compatibility.py`

**Interfaces:**
- Consumes: `ResultEnvelope` from Task 1.
- Produces: `HostInventory.detect(http_get) -> HostInventory`, `HuggingFaceClient.list_models(...)`, `HuggingFaceClient.inspect_model(repo)`, `DiscoveryService.discover(request) -> ResultEnvelope`.
- Preserves: legacy `scout.py --role`, `--limit`, `--new`, `--fast`, `--json`, `--wire`, `--target`, and `--port` behavior.

- [ ] **Step 1: Write classification and compatibility tests using fixed fixtures**

Create tests that copy representative current responses into Python dictionaries, inject them into `DiscoveryService`, and assert quant deduplication, reasoning precedence, actual-byte sizing, and the legacy JSON keys. Add a subprocess test:

```python
def test_legacy_script_and_new_cli_report_same_repositories(self):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "MLX_AGENT_FIXTURE": str(FIXTURE)}
    old = subprocess.run([sys.executable, str(LEGACY), "--json", "--limit", "2"], env=env, text=True, capture_output=True, check=True)
    new = subprocess.run([sys.executable, "-m", "mlx_agent", "discover", "--json", "--limit", "2"], env=env, text=True, capture_output=True, check=True)
    self.assertEqual(repo_ids(json.loads(old.stdout)), repo_ids(json.loads(new.stdout)["data"]))
```

- [ ] **Step 2: Run tests and verify missing-module failures**

Run: `PYTHONPATH=src python3 -m unittest tests.unit.test_models tests.integration.test_scout_compatibility -v`

Expected: FAIL because the extracted modules and new CLI do not exist.

- [ ] **Step 3: Move logic without changing ranking semantics**

Move constants and pure functions from `scout.py` into `models.py`; host probes into `host.py`; HTTP access into `huggingface.py`; and report construction into `discovery.py`. Use dependency injection for subprocess and HTTP calls so tests never require the network.

Implement `cli.main(argv=None) -> int` with the `discover` subcommand and versioned JSON envelopes. Make `skills/mlx-scout/scripts/scout.py` a compatibility wrapper that resolves repository `src`, imports `mlx_agent.cli.legacy_scout_main`, and preserves old output.

- [ ] **Step 4: Prove old and new surfaces**

Run: `PYTHONPATH=src python3 -m unittest tests.unit.test_models tests.integration.test_scout_compatibility -v`

Expected: all tests pass.

Run: `PYTHONPATH=src python3 -m mlx_agent discover --fast --limit 1`

Expected: human-readable host and role output, or a classified `network_unavailable` error with remediation.

- [ ] **Step 5: Commit the extraction**

```bash
git add src/mlx_agent scripts/mlx-agent skills/mlx-scout/scripts/scout.py tests
git commit -m "refactor: extract structured mlx scout core"
```

### Task 3: Add inventory, cache, filters, and ranking explanations

**Files:**
- Modify: `src/mlx_agent/host.py`
- Modify: `src/mlx_agent/huggingface.py`
- Modify: `src/mlx_agent/models.py`
- Modify: `src/mlx_agent/discovery.py`
- Modify: `src/mlx_agent/cli.py`
- Create: `tests/unit/test_filters.py`
- Create: `tests/integration/test_discovery_cache.py`

**Interfaces:**
- Produces: `DiscoveryRequest(role, memory_gb, quantization, licenses, include_gated, publishers, runtime, refresh, offline, limit)`.
- Produces: candidates with `facts`, `estimates`, `heuristics`, `provenance`, `rank_score`, `selection_reasons`, and `rejection_reasons`.
- Produces: cache entries `{schema_version, fetched_at, request, response}` stored under injected `state_dir`.

- [ ] **Step 1: Write failing filter and offline-cache tests**

Cover: license exclusion, gated exclusion, memory budget, runtime compatibility, publisher allow-list, quantization, explicit refresh bypass, fresh-cache reuse, stale-cache warning, and offline failure when no cache exists. Assert stable error code `offline_cache_missing` and warning code `stale_cache`.

- [ ] **Step 2: Verify failures**

Run: `PYTHONPATH=src python3 -m unittest tests.unit.test_filters tests.integration.test_discovery_cache -v`

Expected: FAIL because `DiscoveryRequest` and cache behavior are absent.

- [ ] **Step 3: Implement filters, provenance, and atomic cache writes**

Use `tempfile.NamedTemporaryFile` in the destination directory followed by `os.replace`. Cache TTL defaults to 24 hours and is visible in output. Keep ranking deterministic by sorting on a tuple ending in repository ID. Never label an estimate as measured.

- [ ] **Step 4: Run tests and inspect bounded JSON**

Run: `PYTHONPATH=src python3 -m unittest tests.unit.test_filters tests.integration.test_discovery_cache -v`

Expected: all tests pass.

Run: `PYTHONPATH=src python3 -m mlx_agent discover --json --offline --limit 2`

Expected: a versioned result no larger than two candidates per requested role, or `offline_cache_missing` with no traceback.

- [ ] **Step 5: Commit Scout improvements**

```bash
git add src/mlx_agent tests/unit/test_filters.py tests/integration/test_discovery_cache.py
git commit -m "feat: add explainable cached model discovery"
```

### Task 4: Implement verification and resumable Adopt

**Files:**
- Create: `src/mlx_agent/verification.py`
- Create: `src/mlx_agent/adoption.py`
- Modify: `src/mlx_agent/cli.py`
- Create: `schemas/adoption-state.schema.json`
- Create: `tests/unit/test_verification.py`
- Create: `tests/integration/test_adoption.py`

**Interfaces:**
- Produces: `Verifier.verify(candidate, host, allow_network=True) -> VerificationEvidence`.
- Produces: `AdoptionWorkflow.start(request)`, `.advance(state)`, and `.resume(path)`.
- Produces: phases `inspect`, `discover`, `shortlist`, `verify`, `compare`, `recommend`, `complete`.
- Produces: evidence strength enum `runtime_tested`, `runtime_inventory`, `metadata_only`, `heuristic_only`.

- [ ] **Step 1: Write failing state-transition and safety tests**

Use fake runtime clients. Assert that only installed models receive a generation request; missing models receive metadata inspection only; verification concurrency never exceeds `min(4, max(1, ram_gb // 16))`; a saved state resumes at the first incomplete phase; and recommendations reject confirmed reasoners for fast/general utility roles.

- [ ] **Step 2: Verify failures**

Run: `PYTHONPATH=src python3 -m unittest tests.unit.test_verification tests.integration.test_adoption -v`

Expected: FAIL because the verifier, workflow, and state schema are missing.

- [ ] **Step 3: Implement bounded verification and state persistence**

Represent workflow state as a dataclass serialized through the adoption schema. Persist after every completed phase using atomic replacement. Use `ThreadPoolExecutor(max_workers=calculated_limit)` only for independent verification records. Catch per-candidate failures into evidence rather than aborting the entire run.

Add `adopt start`, `adopt resume`, and `adopt status` CLI commands. Default state paths are supplied by the installer/provider scope, while `--state PATH` is always available for direct use.

- [ ] **Step 4: Prove resume and evidence semantics**

Run: `PYTHONPATH=src python3 -m unittest tests.unit.test_verification tests.integration.test_adoption -v`

Expected: all tests pass with no model downloads or live runtime requirements.

- [ ] **Step 5: Commit Adopt core**

```bash
git add src/mlx_agent/verification.py src/mlx_agent/adoption.py src/mlx_agent/cli.py schemas/adoption-state.schema.json tests
git commit -m "feat: add resumable model adoption workflow"
```

### Task 5: Implement transactional Wire rendering and rollback

**Files:**
- Create: `src/mlx_agent/wiring.py`
- Create: `src/mlx_agent/transactions.py`
- Modify: `src/mlx_agent/cli.py`
- Create: `schemas/receipt.schema.json`
- Create: `tests/unit/test_wiring.py`
- Create: `tests/integration/test_transactions.py`

**Interfaces:**
- Produces: `ConfigAdapter.detect(path)`, `.render(model, runtime, existing)`, `.validate(content)`, and `.health_check(endpoint)`.
- Produces: `Transaction.preview(changes)`, `.apply(confirmation) -> Receipt`, and `rollback(receipt_path) -> Receipt`.
- Receipt fields: schema version, transaction ID, adapter version, timestamp, targets, before/after hashes, backup paths, validations, and status.

- [ ] **Step 1: Write failing render and rollback tests**

Cover all five existing runtime targets. Assert deterministic rendering, parse validation, unified diff preview, refusal without confirmation, backup creation, atomic apply, automatic rollback after health-check failure, exact manual rollback, and redaction of values matching keys such as `api_key`, `token`, `secret`, and `authorization`.

- [ ] **Step 2: Verify failures**

Run: `PYTHONPATH=src python3 -m unittest tests.unit.test_wiring tests.integration.test_transactions -v`

Expected: FAIL because adapters and receipts do not exist.

- [ ] **Step 3: Implement transaction protocol**

Use SHA-256 hashes, mode-preserving backups, `tempfile` staging, `os.replace`, and `difflib.unified_diff`. Refuse symlink targets by default. Validate staged content before replacement and validate the live path after replacement. Store only redacted previews and hashes in receipts.

Expose `wire render`, `wire apply --confirm`, `wire status`, and `wire rollback RECEIPT` commands. Preserve legacy `--wire` as a render-only compatibility alias.

- [ ] **Step 4: Prove mutation safety**

Run: `PYTHONPATH=src python3 -m unittest tests.unit.test_wiring tests.integration.test_transactions -v`

Expected: all tests pass, including byte-for-byte restoration after induced validation failure.

- [ ] **Step 5: Commit transactional Wire**

```bash
git add src/mlx_agent/wiring.py src/mlx_agent/transactions.py src/mlx_agent/cli.py schemas/receipt.schema.json tests
git commit -m "feat: make model wiring transactional"
```

### Task 6: Build the universal installer and provider registry

**Files:**
- Create: `src/mlx_agent/providers.py`
- Create: `src/mlx_agent/installer.py`
- Modify: `src/mlx_agent/cli.py`
- Create: `tests/unit/test_providers.py`
- Create: `tests/integration/test_installer.py`

**Interfaces:**
- Produces: `ProviderDefinition(id, detect_commands, user_root, project_root, artifacts, config_paths)`.
- Produces: `detect_providers(env, path) -> list[ProviderDetection]`.
- Produces: `Installer.plan(action, providers, scope, project_root) -> InstallPlan` and `.execute(plan, confirmed) -> Receipt`.

- [ ] **Step 1: Write failing path and round-trip tests**

Create isolated fake homes and projects for all providers. Assert provider detection, exact scope resolution, multi-provider installation, idempotent reinstall, versioned update, uninstall of receipt-owned files only, preservation of user-modified artifacts with a conflict error, and a successful `doctor` result after clean install.

- [ ] **Step 2: Verify failures**

Run: `PYTHONPATH=src python3 -m unittest tests.unit.test_providers tests.integration.test_installer -v`

Expected: FAIL because provider definitions and installer operations are absent.

- [ ] **Step 3: Implement provider detection and receipt-owned installs**

Provider roots must be data-driven from `plugin.json`. Inject home, config root, executable lookup, and project root into tests. The installer copies only declared artifacts, records their hashes, and delegates mutations to `Transaction`. If no provider is named, return detected choices in human and JSON output; do not silently select all.

Expose `providers`, `install`, `update`, `uninstall`, and `doctor` CLI commands with `--scope user|project`, `--project PATH`, `--dry-run`, and `--confirm`.

- [ ] **Step 4: Run all installer round trips**

Run: `PYTHONPATH=src python3 -m unittest tests.unit.test_providers tests.integration.test_installer -v`

Expected: all tests pass and temporary homes contain no files after uninstall except pre-existing fixtures.

- [ ] **Step 5: Commit installer foundation**

```bash
git add src/mlx_agent/providers.py src/mlx_agent/installer.py src/mlx_agent/cli.py tests
git commit -m "feat: add universal scoped plugin installer"
```

### Task 7: Generate Claude Code and generic AgentSkills adapters

**Files:**
- Create: `scripts/generate_adapters.py`
- Create: `providers/claude/.claude-plugin/plugin.json`
- Create: `providers/claude/.claude-plugin/marketplace.json`
- Create: `providers/claude/commands/mlx-scout.md`
- Create: `providers/claude/commands/mlx-adopt.md`
- Create: `providers/claude/commands/mlx-wire.md`
- Create: `providers/claude/agents/mlx-advisor.md`
- Create: `providers/agentskills/mlx-scout/SKILL.md`
- Create: `providers/agentskills/mlx-adopt/SKILL.md`
- Create: `providers/agentskills/mlx-wire/SKILL.md`
- Modify: `.claude-plugin/plugin.json`
- Modify: `.claude-plugin/marketplace.json`
- Modify: `commands/*.md`
- Modify: `agents/mlx-advisor.md`
- Test: `tests/contracts/test_generated_adapters.py`

**Interfaces:**
- Consumes: canonical `plugin.json` and structured CLI operations.
- Produces: `generate(provider_ids, output_root) -> list[Path]` with deterministic UTF-8/LF output.
- Produces: Claude and AgentSkills prompts that call the same core commands and enforce download/config confirmation gates.

- [ ] **Step 1: Write failing golden and parity tests**

Assert that generation into a temporary directory matches committed provider artifacts byte-for-byte; all prompt files mention the canonical capability ID; every command uses provider-root variables only in the adapter; and no prompt embeds model-ranking logic or asks an agent to edit config directly.

- [ ] **Step 2: Verify failures**

Run: `PYTHONPATH=src python3 -m unittest tests.contracts.test_generated_adapters -v`

Expected: FAIL because the generator and provider outputs are absent.

- [ ] **Step 3: Implement deterministic adapters and migrate legacy paths**

Generate Claude metadata and prompts from the manifest plus small provider templates. Replace the Claude Workflow's embedded business logic with a thin call to `adopt start/resume`; native subagents may fulfill verification work items but must write results through the adoption schema.

Keep current root Claude files as generated compatibility artifacts for existing marketplace users. Generate three generic AgentSkills skills with relative core invocation and no Claude-specific variables.

- [ ] **Step 4: Validate generated drift and current Claude install**

Run: `PYTHONPATH=src python3 scripts/generate_adapters.py --check`

Expected: exit 0 and `generated adapters are current`.

Run: `PYTHONPATH=src python3 -m unittest tests.contracts.test_generated_adapters -v`

Expected: all tests pass.

- [ ] **Step 5: Commit adapter generation**

```bash
git add plugin.json scripts/generate_adapters.py providers/claude providers/agentskills .claude-plugin commands agents scripts/mlx-adopt.workflow.mjs tests/contracts
git commit -m "feat: generate Claude and AgentSkills adapters"
```

### Task 8: Add the native Codex adapter

> **Verified contract supersedes the original guess:** The original plan
> assumed custom `/mlx-*` Codex slash commands. Current verified official and
> installed Codex behavior instead exposes namespaced installed skills:
> `$mlx-agent:mlx-scout`, `$mlx-agent:mlx-adopt`, and `$mlx-agent:mlx-wire`.
> Do not create `providers/codex/commands/*` artifacts.

**Files:**
- Create: `providers/codex/.codex-plugin/plugin.json`
- Create: `providers/codex/skills/mlx-scout/SKILL.md`
- Create: `providers/codex/skills/mlx-adopt/SKILL.md`
- Create: `providers/codex/skills/mlx-wire/SKILL.md`
- Modify: `scripts/generate_adapters.py`
- Modify: `src/mlx_agent/providers.py`
- Create: `tests/contracts/test_codex_adapter.py`
- Create: `tests/smoke/codex.sh`

**Interfaces:**
- Consumes: Codex's current public plugin schema and verified native installed-skill contract, captured as validation assertions in `test_codex_adapter.py`.
- Produces: exactly `$mlx-agent:mlx-scout`, `$mlx-agent:mlx-adopt`, and `$mlx-agent:mlx-wire` after installation, with no `providers/codex/commands/*` surface.
- Produces: Codex-native installed-skill instructions with sequential Adopt fallback.

- [ ] **Step 1: Encode the current Codex public contract as failing tests**

Add fixture-free structural assertions for `.codex-plugin/plugin.json`, the three namespaced installed-skill invocations, absence of `providers/codex/commands/*`, skill directory names, and provider-relative root substitution. Add an installer test proving user and project destinations resolved by `ProviderDefinition("codex", ...)` match the current Codex contract.

- [ ] **Step 2: Verify failures**

Run: `PYTHONPATH=src python3 -m unittest tests.contracts.test_codex_adapter -v`

Expected: FAIL because Codex artifacts are absent.

- [ ] **Step 3: Generate the native Codex package**

Implement the adapter using the provider's documented `.codex-plugin/plugin.json` and `skills/` fields. The installed skills call `discover`, `adopt start/resume`, and `wire render/apply`, preserve confirmation gates, request bounded JSON, and render concise results. Do not copy Claude Workflow syntax or recreate unsupported slash-command artifacts.

- [ ] **Step 4: Validate and smoke-test when Codex is present**

Run: `PYTHONPATH=src python3 scripts/generate_adapters.py --check && PYTHONPATH=src python3 -m unittest tests.contracts.test_codex_adapter -v`

Expected: all tests pass.

Run: `bash tests/smoke/codex.sh`

Expected: SKIP with exit 0 when Codex is unavailable; otherwise install into a temporary Codex home, use `codex debug prompt-input` to discover `$mlx-agent:mlx-scout`, `$mlx-agent:mlx-adopt`, and `$mlx-agent:mlx-wire`, invoke `$mlx-agent:mlx-scout` with a fixture-only noninteractive session, uninstall, and exit 0.

- [ ] **Step 5: Commit Codex support**

```bash
git add providers/codex scripts/generate_adapters.py src/mlx_agent/providers.py tests/contracts/test_codex_adapter.py tests/smoke/codex.sh
git commit -m "feat: add native Codex plugin adapter"
```

### Task 9: Add the native Gemini CLI extension

**Files:**
- Create: `providers/gemini/gemini-extension.json`
- Create: `providers/gemini/commands/mlx-scout.toml`
- Create: `providers/gemini/commands/mlx-adopt.toml`
- Create: `providers/gemini/commands/mlx-wire.toml`
- Create: `providers/gemini/skills/mlx-scout/SKILL.md`
- Create: `providers/gemini/skills/mlx-adopt/SKILL.md`
- Create: `providers/gemini/skills/mlx-wire/SKILL.md`
- Modify: `scripts/generate_adapters.py`
- Modify: `src/mlx_agent/providers.py`
- Create: `tests/contracts/test_gemini_adapter.py`
- Create: `tests/smoke/gemini.sh`

**Interfaces:**
- Consumes: Gemini CLI extension and custom-command contracts from `https://geminicli.com/docs/extensions/reference/`.
- Produces: an installable local/GitHub extension exposing the three native commands.

- [ ] **Step 1: Write failing manifest, command, and scope tests**

Assert required extension keys, command TOML parsing with `tomllib` on Python 3.11 and a small fallback parser on Python 3.9/3.10, absence of absolute repository paths, and exact command parity.

- [ ] **Step 2: Verify failures**

Run: `PYTHONPATH=src python3 -m unittest tests.contracts.test_gemini_adapter -v`

Expected: FAIL because Gemini artifacts are absent.

- [ ] **Step 3: Generate the Gemini extension**

Generate `gemini-extension.json`, native command TOML files, and skills from the canonical contract. Adopt uses Gemini's native orchestration where exposed and otherwise calls the sequential resumable core. Register user/project install paths in `providers.py`.

- [ ] **Step 4: Validate and smoke-test when Gemini is present**

Run: `PYTHONPATH=src python3 scripts/generate_adapters.py --check && PYTHONPATH=src python3 -m unittest tests.contracts.test_gemini_adapter -v`

Expected: all tests pass.

Run: `bash tests/smoke/gemini.sh`

Expected: SKIP with exit 0 when Gemini CLI is unavailable; otherwise install from the local provider directory into a temporary home, verify all commands with `gemini extensions`, uninstall, and exit 0.

- [ ] **Step 5: Commit Gemini support**

```bash
git add providers/gemini scripts/generate_adapters.py src/mlx_agent/providers.py tests/contracts/test_gemini_adapter.py tests/smoke/gemini.sh
git commit -m "feat: add native Gemini CLI extension"
```

### Task 10: Add the native OpenCode package

**Files:**
- Create: `providers/opencode/opencode.json`
- Create: `providers/opencode/commands/mlx-scout.md`
- Create: `providers/opencode/commands/mlx-adopt.md`
- Create: `providers/opencode/commands/mlx-wire.md`
- Create: `providers/opencode/agents/mlx-advisor.md`
- Create: `providers/opencode/skills/mlx-scout/SKILL.md`
- Create: `providers/opencode/skills/mlx-adopt/SKILL.md`
- Create: `providers/opencode/skills/mlx-wire/SKILL.md`
- Modify: `scripts/generate_adapters.py`
- Modify: `src/mlx_agent/providers.py`
- Create: `tests/contracts/test_opencode_adapter.py`
- Create: `tests/smoke/opencode.sh`

**Interfaces:**
- Consumes: OpenCode command, agent, skill, and plugin contracts from `https://opencode.ai/docs/commands/`, `/agents/`, and `/plugins/`.
- Produces: three native commands and an advisor agent; Adopt may set `subtask: true` for bounded verification work.

- [ ] **Step 1: Write failing OpenCode contract tests**

Assert valid config JSON, native Markdown command frontmatter, exact command parity, advisor permissions that require confirmation for edits/bash mutations, and both user/project destination mappings.

- [ ] **Step 2: Verify failures**

Run: `PYTHONPATH=src python3 -m unittest tests.contracts.test_opencode_adapter -v`

Expected: FAIL because OpenCode artifacts are absent.

- [ ] **Step 3: Generate commands, skills, and agent**

Generate OpenCode-native artifacts from the canonical contract. Configure Adopt subtasks only for independent verification records, never for unbounded fan-out. Wire routes every mutation through the transaction CLI and cannot receive blanket edit permission.

- [ ] **Step 4: Validate and smoke-test when OpenCode is present**

Run: `PYTHONPATH=src python3 scripts/generate_adapters.py --check && PYTHONPATH=src python3 -m unittest tests.contracts.test_opencode_adapter -v`

Expected: all tests pass.

Run: `bash tests/smoke/opencode.sh`

Expected: SKIP with exit 0 when OpenCode is unavailable; otherwise use an isolated config directory, list and invoke the three commands against fixtures, uninstall, and exit 0.

- [ ] **Step 5: Commit OpenCode support**

```bash
git add providers/opencode scripts/generate_adapters.py src/mlx_agent/providers.py tests/contracts/test_opencode_adapter.py tests/smoke/opencode.sh
git commit -m "feat: add native OpenCode plugin adapter"
```

### Task 11: Complete documentation, compatibility evidence, and CI gates

**Files:**
- Modify: `README.md`
- Create: `docs/install/index.md`
- Create: `docs/install/claude.md`
- Create: `docs/install/codex.md`
- Create: `docs/install/gemini.md`
- Create: `docs/install/opencode.md`
- Create: `docs/guides/scout.md`
- Create: `docs/guides/adopt.md`
- Create: `docs/guides/wire.md`
- Create: `docs/security.md`
- Create: `docs/adding-a-provider.md`
- Create: `docs/migrating-from-v0.1.md`
- Create: `compatibility/providers.json`
- Create: `tests/contracts/test_docs.py`
- Create: `.github/workflows/test.yml`
- Create: `.github/workflows/provider-smoke.yml`
- Create: `.github/workflows/apple-silicon.yml`

**Interfaces:**
- Consumes: native install commands proven by Tasks 7-10 and provider smoke results.
- Produces: user-visible support claims generated from `compatibility/providers.json`.
- Produces: portable PR gates and explicit Apple Silicon release evidence.

- [ ] **Step 1: Write failing documentation-integrity and compatibility tests**

Assert every supported provider has native and universal install instructions, update/uninstall/doctor commands, exact Scout/Adopt/Wire names, minimum/last-tested versions, scope support, config paths, last smoke-test status/date, and links from README. Assert no docs claim a provider/version absent from the matrix.

- [ ] **Step 2: Verify failures**

Run: `PYTHONPATH=src python3 -m unittest tests.contracts.test_docs -v`

Expected: FAIL because the matrix and new documentation are absent.

- [ ] **Step 3: Write goal-oriented docs and evidence matrix**

Make README start with a provider selector and universal installer example. Document dry-run/confirmation behavior, state locations, receipt recovery, offline Scout, Adopt evidence levels, Wire rollback, and migration from the existing Claude marketplace installation. Populate compatibility entries only with versions actually validated during implementation; use status `not-run` rather than inventing proof.

- [ ] **Step 4: Add CI workflows with explicit responsibilities**

`test.yml` runs compile checks, all `unittest` suites, adapter drift checks, contract validation, install/update/uninstall round trips, and `git diff --check` on macOS and Linux.

`provider-smoke.yml` runs manually and on a schedule, installs only CLIs whose automation and licensing permit it, uses isolated homes, and uploads non-secret smoke results.

`apple-silicon.yml` targets a self-hosted Apple Silicon label, runs host/runtime inventory and fixture-backed verification on PRs, and requires a live supported-runtime health check for release tags.

- [ ] **Step 5: Run the complete local release gate**

Run: `PYTHONPATH=src python3 -m compileall -q src skills scripts tests`

Expected: exit 0.

Run: `PYTHONPATH=src python3 -m unittest discover -s tests -v`

Expected: all tests pass; unavailable provider CLI smoke tests report SKIP, not PASS.

Run: `PYTHONPATH=src python3 scripts/validate_contracts.py && PYTHONPATH=src python3 scripts/generate_adapters.py --check && git diff --check`

Expected: all commands exit 0 with no generated drift or whitespace errors.

- [ ] **Step 6: Commit docs and release gates**

```bash
git add README.md docs compatibility tests/contracts/test_docs.py .github/workflows
git commit -m "docs: publish universal plugin compatibility and install guides"
```

### Task 12: Perform release-candidate validation without publishing

**Files:**
- Modify: `compatibility/providers.json`
- Create: `docs/release-checklist.md`
- Modify only if generated: provider artifacts and generated README compatibility block

**Interfaces:**
- Consumes: all unit, contract, round-trip, provider CLI, and Apple Silicon checks.
- Produces: a reviewable release-candidate evidence snapshot; does not tag, publish, push, create a release, or change repository visibility.

- [ ] **Step 1: Run each available native provider install round trip in isolated state**

Run each `tests/smoke/<provider>.sh` script. Record exact CLI version, scope, command discovery result, fixture-backed Scout result, uninstall result, date, and environment in `compatibility/providers.json`. Leave unavailable providers as `not-run` and do not upgrade them to supported based on schema tests alone.

- [ ] **Step 2: Run an Apple Silicon live-path check**

Run `inspect-host`, a bounded live `discover --limit 1`, inventory-only verification for an installed model, `adopt start` with a temporary state path, and `wire render` without apply. Confirm no model download and no provider config mutation occurred.

- [ ] **Step 3: Exercise one reversible configuration transaction**

Against a temporary runtime/provider fixture, run preview, confirmed apply, parse validation, health check, status, and rollback. Verify the final fixture hash equals its initial hash and the receipt contains no secret values.

- [ ] **Step 4: Run final gates from a clean worktree**

Run: `PYTHONPATH=src python3 -m unittest discover -s tests -v`

Run: `PYTHONPATH=src python3 scripts/validate_contracts.py`

Run: `PYTHONPATH=src python3 scripts/generate_adapters.py --check`

Run: `git diff --check && git status --short`

Expected: tests and validators pass; generated artifacts are current; status contains only the intended compatibility evidence and release-checklist changes.

- [ ] **Step 5: Commit release-candidate evidence**

```bash
git add compatibility/providers.json docs/release-checklist.md
git commit -m "test: record universal plugin release evidence"
```

Stop after this commit and request explicit approval before any push, tag, marketplace submission, package publication, release creation, or visibility change.
