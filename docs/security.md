# Security and recovery boundaries

The universal installer and Wire are deliberately confirmation-gated. `--dry-run` produces a plan; install, update, uninstall, and Wire apply require a current reviewed `--preview-hash`. The project never auto-installs a provider CLI, auto-downloads a model, or persists secrets.

## Tool-use verification boundary

Tool-use verification inventories only already-running loopback runtimes and already-installed models. It does not pull, install, or download models, and it never starts a runtime. The probe sends one fixed, bounded synthetic tool definition and asks for a synthetic argument; no real user tool executes. Ollama and local OpenAI-compatible LM Studio, `mlx_lm`, and LiteLLM servers support this probe. `mlx-vlm` produces `unsupported-runtime` evidence for tool-use.

The opt-in release live smoke automatically selects only direct local Ollama, LM Studio, or `mlx_lm` backends. It excludes LiteLLM from auto-selection because a loopback LiteLLM endpoint can proxy a remote or paid backend. A LiteLLM smoke is appropriate only when its configured route has been independently proven local.

Only the normalized outcome and evidence labels are retained. Raw prompt, response, endpoint, and credentials are not persisted. Runtime and metadata errors are bounded and redacted before serialization.

Installer receipts live at `~/mlx-agent/installer-receipts` by default. `$MLX_AGENT_CONFIG_ROOT` takes precedence when set; otherwise `$XDG_STATE_HOME` relocates them to `$XDG_STATE_HOME/mlx-agent/installer-receipts`. Project scope uses `<project>/.mlx-agent/installer-receipts`. Wire receipts default to `.mlx-agent-receipts` beside the changed config unless `--receipts-dir` is supplied. Receipts contain paths, hashes, backups, validation results, and recovery state; they are not credential stores.

Run `python3 scripts/mlx-agent doctor <provider> --scope user --json` to check receipt ownership. If it reports `batch_recovery_required`, preserve the files and receipt, inspect the listed child receipt, then use the documented uninstall or Wire rollback command only after reviewing the hashes. The installer refuses to overwrite or delete unowned or user-modified artifacts.

## Installer scoped-lock upgrade

Installer transactions hold their locks under scoped installer receipt storage, not beside installed artifacts. During the serialized upgrade window, the installer safely opens each physical target parent and descriptor-relatively enumerates only strict legacy names of the form `.mlx-agent-wire-<64 lowercase hex>.lock`. Every such candidate is opened no-follow, validated as a current-user regular file, and acquired nonblocking before any candidate is removed. Upgrade requires all older mlx-agent processes stopped: a busy candidate fails closed as `legacy_lock_busy`, while stale candidates are removed only while exclusively held and the opened parent filesystem identity is added to the versioned `legacy-target-locks-v1` scoped state map.

After a physical parent is migrated, an older binary is unsupported for targets in that parent. Any later strict legacy candidate there makes install/update/rollback fail closed, and `doctor` reports `legacy_lock_recreated` with the affected paths. Invalid lookalike names are never opened or removed, and unmigrated parents can still complete their own safe upgrade. A historical version-only marker is retained as an unscoped fail-closed barrier because its parent cannot be recovered. This parent-scoped policy is intentionally conservative because legacy names contain one-way path-spelling hashes. It guards the upgrade boundary; it does not claim to coordinate a future older process that disregards the new lock protocol.

Wire holds an advisory target lock for cooperative writers. Current lock identity uses the opened parent device/inode and an NFC-normalized, case-folded leaf. Python's standard library does not expose a stable descriptor-bound filesystem case-sensitivity capability, so case-distinct leaves conservatively contend even on case-sensitive parents; this safe false contention never weakens macOS alias protection. The lock protects against accidental concurrency, not a malicious process that ignores it and races the final rename. This residual risk is reported in every Wire result and receipt; use a filesystem/permission boundary when adversarial local writers are in scope.
