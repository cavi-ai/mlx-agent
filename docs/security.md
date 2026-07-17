# Security and recovery boundaries

The universal installer and Wire are deliberately confirmation-gated. `--dry-run` produces a plan; install, update, uninstall, and Wire apply require a current reviewed `--preview-hash`. The project never auto-installs a provider CLI, auto-downloads a model, or persists secrets.

Installer receipts live at `~/mlx-agent/installer-receipts` by default, or under `$MLX_AGENT_CONFIG_ROOT/mlx-agent/installer-receipts` when that root is set; project scope uses `<project>/.mlx-agent/installer-receipts`. Wire receipts default to `.mlx-agent-receipts` beside the changed config unless `--receipts-dir` is supplied. Receipts contain paths, hashes, backups, validation results, and recovery state; they are not credential stores.

Run `python3 scripts/mlx-agent doctor <provider> --scope user --json` to check receipt ownership. If it reports `batch_recovery_required`, preserve the files and receipt, inspect the listed child receipt, then use the documented uninstall or Wire rollback command only after reviewing the hashes. The installer refuses to overwrite or delete unowned or user-modified artifacts.

Wire holds an advisory target lock for cooperative writers. It protects against accidental concurrency, not a malicious process that ignores the lock and races the final rename. This residual risk is reported in every Wire result and receipt; use a filesystem/permission boundary when adversarial local writers are in scope.
