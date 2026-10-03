# Install mlx-agent

Choose the host that owns your coding-agent surface: [Claude Code](claude.md), [Codex CLI](codex.md), [Agy](agy.md), or [OpenCode](opencode.md). For any AgentSkills-compatible host, use the universal installer: it copies each skill's `SKILL.md` together with the launcher and runtime from the repository root, so every installed skill folder is self-contained.

For a user-scoped portable install, install all seven skills into the host's AgentSkills directory:

```bash
python3 scripts/mlx-agent install agentskills --scope user --dry-run --json
python3 scripts/mlx-agent install agentskills --scope user --confirm --preview-hash <preview-hash> --json
```

For project scope, add `--scope project --project /absolute/project/path`; skills go to `<project>/.agents/skills`. Restart the host and verify its available-skills list contains `mlx-scout`, `mlx-adopt`, `mlx-wire`, `mlx-bench`, `mlx-doctor`, `mlx-watch`, and `mlx-fleet`. Update and uninstall with the same preview-then-confirm sequence (`update agentskills`, `uninstall agentskills`). A skill folder copied by hand from an earlier release is not receipt-owned, so the installer refuses to overwrite it; move existing `mlx-*` folders out of the skills directory before the first `install agentskills`.

All provider packages run the same structured Python core and require Python 3.9 or later. A package holds one copy of that runtime: the repository root for Claude Code and OpenCode, `providers/codex` and `providers/agy` for their plugin directories, and one per installed skill for AgentSkills. `update` and `uninstall` also remove files an earlier release installed that the current layout no longer declares, but only receipt-owned files whose content is unchanged. The universal installer stages only receipt-owned artifacts; it never installs a provider CLI, downloads model weights, persists secrets, or edits an unowned configuration file. `MLX_AGENT_CONFIG_ROOT` explicitly relocates MLX-agent receipts. When it is unset, `XDG_STATE_HOME` relocates those receipts; when neither is set they default to `~/.local/state/mlx-agent/installer-receipts`. OpenCode additionally follows `XDG_CONFIG_HOME`; other provider user roots remain anchored to the selected host's home directory. A provider directory that is a symlink needs no variables at all: the installer resolves it once and records the resolved path.

```bash
# Run from this repository or an unpacked release.
python3 scripts/mlx-agent providers --json
python3 scripts/mlx-agent install agy --scope user --dry-run --json
```

Inspect the returned `preview.preview_hash`. Only then repeat the operation with the exact hash:

```bash
python3 scripts/mlx-agent install agy --scope user --confirm --preview-hash <preview-hash> --json
```

Use the same preview/confirmation sequence for `update` and `uninstall`. `doctor` is read-only and reports `portable`, `staged`, or `native-visible` integration separately from receipt-owned artifact validity:

```bash
python3 scripts/mlx-agent update agy --scope user --dry-run --json
python3 scripts/mlx-agent uninstall agy --scope user --dry-run --json
python3 scripts/mlx-agent doctor agy --scope user --json
```

The website catalog specializes these safe lifecycle previews for each provider and scope from the following canonical templates:

```bash
python3 scripts/mlx-agent install claude codex agy opencode agentskills --scope user --dry-run --json
python3 scripts/mlx-agent doctor claude codex agy opencode agentskills --scope user --json
python3 scripts/mlx-agent update claude codex agy opencode agentskills --scope user --dry-run --json
python3 scripts/mlx-agent uninstall claude codex agy opencode agentskills --scope user --dry-run --json
```

For a mutating action, inspect the returned preview and rerun the specialized command with `--confirm --preview-hash <preview-hash> --json`.

For project scope, add `--scope project --project /absolute/project/path`. Project receipts stay under `<project>/.mlx-agent/installer-receipts`.
