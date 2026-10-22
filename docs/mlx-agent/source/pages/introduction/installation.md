# Installation

The core requires Python 3.9+ and uses the standard library. MLX execution and
macOS host inspection require Apple Silicon. Install the coding-agent host CLI
separately. Optional model runtimes have their own dependencies; managed optional
backend installation uses Python 3.12 locks.

These instructions use the package version recorded in this checkout. To work
with a published release, select its tag and documentation together. To run the
CLI against the current source, clone the default branch and run commands from
the repository root:

```bash
git clone https://github.com/cavi-ai/mlx-agent.git
cd mlx-agent
python3 scripts/mlx-agent --help
python3 scripts/mlx-agent inspect-host --json
```

Choose a host below, then follow [Your first local model](quickstart.md).

## Claude Code

```bash
claude plugin marketplace add cavi-ai/mlx-agent
claude plugin install mlx-agent@mlx-agent
```

Restart Claude Code, then run `/mlx-scout`, `/mlx-adopt`, `/mlx-wire`, `/mlx-bench`, `/mlx-doctor`, `/mlx-watch`, or `/mlx-fleet`.

## Codex CLI

```bash
codex plugin marketplace add cavi-ai/mlx-agent --ref v0.10.0
codex plugin add mlx-agent@mlx-agent
```

Restart Codex, then invoke `$mlx-agent:mlx-scout` and the other skills. Codex does not support custom slash commands.

## Agy

```bash
git clone --depth 1 --branch v0.10.0 https://github.com/cavi-ai/mlx-agent.git
agy plugin validate ./mlx-agent/providers/agy
agy plugin install ./mlx-agent/providers/agy
```

Restart Agy, use `/skills` to confirm the package, then ask it to use `mlx-scout` or another packaged skill.

## OpenCode

```bash
git clone --depth 1 --branch v0.10.0 https://github.com/cavi-ai/mlx-agent.git
cd mlx-agent
python3 scripts/mlx-agent install opencode --scope user --dry-run --json
# Review the plan; copy data.preview.preview_hash.
python3 scripts/mlx-agent install opencode --scope user --confirm --preview-hash <preview-hash> --json
python3 scripts/mlx-agent doctor opencode --scope user --json
```

Restart OpenCode, press `Ctrl+P`, filter for `mlx`, and run `/mlx-scout`.
Keep any custom XDG paths consistent between installation and host launch.

## OpenClaw

From a checkout, with the gateway's `OPENCLAW_STATE_DIR` exported when it is not
`~/.openclaw`:

```bash
python3 scripts/mlx-agent install openclaw --scope user --dry-run --json
# Review the plan; copy data.preview.preview_hash.
python3 scripts/mlx-agent install openclaw --scope user --confirm --preview-hash <preview-hash> --json
```

Start a new session, check `openclaw skills list`, and run `/mlx-scout`.

## Hermes Agent

From a checkout, with `HERMES_HOME` set to the profile directory when it is not
`~/.hermes`:

```bash
python3 scripts/mlx-agent install hermes --scope user --dry-run --json
# Review the plan; copy data.preview.preview_hash.
python3 scripts/mlx-agent install hermes --scope user --confirm --preview-hash <preview-hash> --json
```

Check `hermes skills list`, then run `/mlx-scout`. Project installs go to
`<project>/.hermes/skills`; Hermes releases with project skill discovery load
them after `hermes skills trust`.

## AgentSkills-compatible hosts

From a checkout:

```bash
python3 scripts/mlx-agent install agentskills --scope user --dry-run --json
# Review the plan; copy data.preview.preview_hash.
python3 scripts/mlx-agent install agentskills --scope user --confirm --preview-hash <preview-hash> --json
```

Restart the host and check its skill list. The installer copies a launcher and
runtime into each installed skill so it can operate on its own. Generated skill
definitions copied by hand from the repository do not include that runtime.
Move unowned older `mlx-*` skill folders aside before the first managed install.

## Universal installer

The preview-then-confirm installer covers `claude`, `codex`, `agy`, `opencode`,
`openclaw`, `hermes`, and `agentskills` in user or project scope:

```bash
python3 scripts/mlx-agent providers --json
python3 scripts/mlx-agent install agy --scope user --dry-run --json
# Review the plan; copy data.preview.preview_hash.
python3 scripts/mlx-agent install agy --scope user --confirm --preview-hash <preview-hash> --json
```

For project installs, use `--scope project --project /absolute/project/path`
for both preview and confirmation. Use the same sequence for `update` and
`uninstall`; run `doctor <provider>` with matching scope to inspect ownership.
Installer changes are receipt-owned; unowned or user-modified artifacts are
refused. Installing an adapter does not download model weights.

The [provider lifecycle guide](https://github.com/cavi-ai/mlx-agent/blob/main/docs/install/index.md)
includes host-specific configuration paths and recovery. See
[Backends](../lifecycle/backends.md) for explicit dependency installation and
[Runtimes](../reference/runtimes.md) for model execution choices.
