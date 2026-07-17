#!/usr/bin/env bash
# Smoke the native Gemini extension lifecycle without touching the caller's home.
set -euo pipefail

if ! command -v gemini >/dev/null 2>&1; then
  echo "SKIP: Gemini CLI unavailable"
  exit 0
fi

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
SMOKE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mlx-agent-gemini-smoke.XXXXXX")"
cleanup() {
  rm -rf "$SMOKE_ROOT"
}
trap cleanup EXIT

export HOME="$SMOKE_ROOT/home"
export XDG_CONFIG_HOME="$SMOKE_ROOT/config"
export XDG_CACHE_HOME="$SMOKE_ROOT/cache"
mkdir -p "$HOME" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"

assert_discovered_commands() {
  command_output="$1"
  scope_label="$2"
  for command in mlx-scout mlx-adopt mlx-wire; do
    case "$command_output" in
      *"$command.toml"*) ;;
      *) echo "Gemini did not list $scope_label custom command file: $command.toml" >&2; exit 1 ;;
    esac
  done
}

list_custom_commands() {
  command_cwd="$1"
  # `/commands list` is Gemini's documented command-discovery surface. Gemini
  # 0.46 prints the local list, then can continue into model routing. Require
  # an explicit opt-in before invoking it, rather than claiming it is a
  # network-free CLI operation.
  (
    cd "$command_cwd"
    gemini -p '/commands list' --output-format text 2>&1 || true
  )
}

assert_skills() {
  skills="$1"
  scope_label="$2"
  for capability in mlx-scout mlx-adopt mlx-wire; do
    case "$skills" in
      *"$capability"*) ;;
      *) echo "Gemini did not discover $scope_label bundled skill: $capability" >&2; exit 1 ;;
    esac
  done
}

# Gemini validates command TOML and the extension structure before installation.
# In particular, commands/mlx-scout.toml exposes /mlx-scout (with matching
# /mlx-adopt and /mlx-wire command files).
gemini extensions validate "$ROOT/providers/gemini"

# Gemini 0.46 currently asks for source, workspace, and skills consent; newer
# releases may split those checks further. The bounded answers are confined to
# the disposable home; no caller settings, secrets, or login state are read or
# copied. Disable pipefail just for `yes`: Gemini's exit code remains the
# pipeline status, while `yes` normally ends with SIGPIPE after install.
set +o pipefail
yes y | gemini extensions install "$ROOT/providers/gemini"
INSTALL_STATUS=$?
set -o pipefail
if [ "$INSTALL_STATUS" -ne 0 ]; then
  exit "$INSTALL_STATUS"
fi

EXTENSIONS="$(gemini extensions list 2>&1)"
case "$EXTENSIONS" in
  *"mlx-agent"*) ;;
  *) echo "Gemini did not discover the installed mlx-agent extension" >&2; exit 1 ;;
esac

# `/commands list` is the official custom-command discovery surface. Gemini
# 0.46 lists extension filenames, which map to the exact slash commands
# `/mlx-scout`, `/mlx-adopt`, and `/mlx-wire` by the documented TOML filename
# convention. It is intentionally opt-in because the current CLI may route
# the prompt after printing that local list.
if [ "${MLX_AGENT_GEMINI_LIVE_COMMAND_DISCOVERY:-}" = "1" ]; then
  USER_COMMANDS="$(list_custom_commands "$ROOT")"
  assert_discovered_commands "$USER_COMMANDS" "user-scope"
else
  echo 'SKIP: set MLX_AGENT_GEMINI_LIVE_COMMAND_DISCOVERY=1 to probe /commands list'
fi
USER_SKILLS="$(gemini skills list --all 2>&1)"
assert_skills "$USER_SKILLS" "user-scope"

# Keep the direct launcher check as a separate self-contained bundle proof; it
# does not substitute for Gemini extension, command-TOML validation, or skill discovery.
MLX_AGENT_FIXTURE="$ROOT/tests/fixtures/scout_responses.json" \
  python3 "$HOME/.gemini/extensions/mlx-agent/skills/mlx-scout/scripts/mlx-agent" \
    discover --limit 1 --json >/dev/null

gemini extensions uninstall mlx-agent
REMOVED="$(gemini extensions list 2>&1)"
case "$REMOVED" in
  *"mlx-agent"*) echo "Gemini did not remove mlx-agent from the isolated home" >&2; exit 1 ;;
esac

# Project scope is installed by the structured installer, not by modifying a
# Gemini settings file. Start Gemini from that disposable project and prove the
# extension's documented command and skill discovery surfaces independently.
PROJECT="$SMOKE_ROOT/project"
mkdir -p "$PROJECT"
PROJECT_PLAN="$(python3 "$ROOT/scripts/mlx-agent" install gemini --scope project --project "$PROJECT" --dry-run --json)"
PROJECT_HASH="$(printf '%s' "$PROJECT_PLAN" | python3 -c 'import json, sys; print(json.loads(sys.stdin.read())["data"]["preview"]["preview_hash"])')"
python3 "$ROOT/scripts/mlx-agent" install gemini --scope project --project "$PROJECT" --confirm --preview-hash "$PROJECT_HASH" --json >/dev/null
test -f "$PROJECT/.gemini/extensions/mlx-agent/gemini-extension.json"
if [ "${MLX_AGENT_GEMINI_LIVE_COMMAND_DISCOVERY:-}" = "1" ]; then
  PROJECT_COMMANDS="$(list_custom_commands "$PROJECT")"
  assert_discovered_commands "$PROJECT_COMMANDS" "project-scope"
fi
# Workspace skills are intentionally gated on workspace trust. The disposable
# project is trusted for this one read-only discovery call only.
PROJECT_SKILLS="$(cd "$PROJECT" && GEMINI_CLI_TRUST_WORKSPACE=true gemini skills list --all 2>&1)"
assert_skills "$PROJECT_SKILLS" "project-scope"
PROJECT_REMOVE="$(python3 "$ROOT/scripts/mlx-agent" uninstall gemini --scope project --project "$PROJECT" --dry-run --json)"
PROJECT_REMOVE_HASH="$(printf '%s' "$PROJECT_REMOVE" | python3 -c 'import json, sys; print(json.loads(sys.stdin.read())["data"]["preview"]["preview_hash"])')"
python3 "$ROOT/scripts/mlx-agent" uninstall gemini --scope project --project "$PROJECT" --confirm --preview-hash "$PROJECT_REMOVE_HASH" --json >/dev/null
test ! -e "$PROJECT/.gemini/extensions/mlx-agent/gemini-extension.json"
for capability in mlx-scout mlx-adopt mlx-wire; do
  test ! -e "$PROJECT/.gemini/commands/$capability.toml"
  test ! -e "$PROJECT/.gemini/skills/$capability/SKILL.md"
done

echo 'PASS: Gemini user and project extension scopes validated, discovered skills, proved the installed bundle, and removed cleanly'
