# Release-candidate validation checklist

Date: 2026-07-17

This evidence was collected without publishing, tagging, pushing, creating a release, changing visibility, downloading a model, or changing a real provider configuration. Provider commands used disposable homes/configuration roots.

## Authorized Task 12 scope expansion

Task 12 was retargeted by user authorization from evidence-only validation to release-blocking runtime/installer hardening after validation uncovered unsafe legacy lock compatibility. This commit therefore includes transaction lock migration and rollback enforcement, installer doctor diagnostics and cleanup, generated provider runtime bundles, CLI host inspection, contract/integration/smoke coverage, and structured release documentation/evidence. The hardening evidence below is separate from the provider smoke evidence: it proves safe upgrade behavior, while provider records describe host lifecycle and fixture/model limitations.

### legacy-lock migration evidence

- New installer transactions and rollback enumerate every strict legacy lock candidate descriptor-relatively under each opened physical target parent while holding the exclusive migration window.
- All candidates must be current-user regular files and must be acquired nonblocking before any unlink; a busy candidate fails closed as `legacy_lock_busy`, while stale candidates are removed only while exclusively held and the physical parent identity is added to the versioned `legacy-target-locks-v1` state map and receipt version.
- Any strict candidate recreated under a migrated parent fails closed as `legacy_lock_recreated`; invalid lookalike filenames remain untouched and another unmigrated parent remains independently migratable. `doctor` reports both states and remediation.
- Upgrade requires all older mlx-agent processes stopped. After migration, an older binary is unsupported; this check can reject a recreated legacy lock but cannot coordinate a future process that ignores the scoped-lock upgrade.

## Provider smoke evidence

The canonical command results and redacted transaction hashes are in [`compatibility/release-evidence.json`](../compatibility/release-evidence.json). The identifiers below, rather than this narrative, are the release evidence references.

| Provider | Evidence ID | Scope and result |
| --- | --- | --- |
| Claude Code | `rc-2026-07-17-claude` | User-scope marketplace lifecycle and fixture bundle evidence. |
| Codex CLI | `rc-2026-07-17-codex` | User-scope lifecycle cleanup passed; fixture-only model invocation remains DNS-blocked. |
| Gemini CLI | `rc-2026-07-17-gemini` | User/project lifecycle and fixture bundle evidence. |
| OpenCode | `rc-2026-07-17-opencode` | Native smoke not run because OpenCode and Bun were unavailable; fixture lifecycle is separate evidence. |
| AgentSkills | `rc-2026-07-17-agentskills` | User/project fixture lifecycle with scoped installer locks and clean artifact removal. |

## Apple Silicon and local-model path

- Host: macOS 26.5.2 on `arm64`. `PYTHONPATH=src python3 scripts/mlx-agent inspect-host --json` returned a versioned inventory with classified `host_probe_unavailable` and `runtime_probe_unavailable` warnings. The sandbox denied the `sysctl` probes, so chip and RAM were unavailable; Ollama and LM Studio inventory both reported unavailable.
- `PYTHONPATH=src python3 scripts/mlx-agent discover --limit 1 --state-dir <temporary-dir> --json` returned `network_unavailable`: DNS resolution for `huggingface.co` failed. This is a classified retryable environment blocker, not a support claim.
- `adopt start --state <temporary-state> --role coding --shortlist-limit 1 --json` used a temporary state path and returned `adoption_failed` from the same DNS failure.
- Inventory-only verification (`allow_network=False`) made no download attempt and reported no locally installed model because both local runtime inventory probes were unavailable/denied.
- `wire render` was exercised only against a temporary JSON fixture; no apply was used for the live-path check and no real provider config was read or changed.

## Reversible Wire transaction

Against a temporary empty JSON configuration and a temporary localhost HTTP fixture, the validation ran:

1. `wire render` with parse validation.
2. Unconfirmed `wire apply` preview and hash capture.
3. Confirmed `wire apply --confirm --preview-hash ...` with a passing local-only health check.
4. `wire status` receipt inspection.
5. `wire rollback --confirm`.

The final SHA-256 of the temporary config equalled its initial SHA-256. The receipt retained only the local endpoint and no fixture secret value; no model download command was executed.

## Final release gates

Run before the evidence commit:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/validate_contracts.py
PYTHONPATH=src python3 scripts/generate_adapters.py --check
python3 scripts/render_compatibility.py --write
git diff --check
git status --short
```

Release status: blocked by all unresolved gates: (1) a network-enabled Apple Silicon discovery rerun, (2) Codex model-backed invocation blocked by DNS, and (3) OpenCode/Bun native smoke unavailable. Gemini and Claude model sessions were intentionally not attempted.
