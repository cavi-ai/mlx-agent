#!/usr/bin/env bash
# Smoke the native OpenCode package without reading caller credentials or config.
set -euo pipefail

if ! command -v opencode >/dev/null 2>&1; then
  echo "SKIP: OpenCode CLI unavailable"
  exit 0
fi

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
SMOKE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mlx-agent-opencode-smoke.XXXXXX")"
cleanup() {
  rm -rf "$SMOKE_ROOT"
}
trap cleanup EXIT

export HOME="$SMOKE_ROOT/home"
export XDG_CONFIG_HOME="$HOME/.config"
export XDG_CACHE_HOME="$SMOKE_ROOT/cache"
export XDG_DATA_HOME="$SMOKE_ROOT/data"
export OPENCODE_DISABLE_AUTOUPDATE=true
export OPENCODE_DISABLE_MODELS_FETCH=true
mkdir -p "$HOME" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" "$XDG_DATA_HOME"

install_scope() {
  scope="$1"
  project="$2"
  plan="$(python3 "$ROOT/scripts/mlx-agent" install opencode --scope "$scope" --project "$project" --dry-run --json)"
  preview_hash="$(printf '%s' "$plan" | python3 -c 'import json, sys; print(json.loads(sys.stdin.read())["data"]["preview"]["preview_hash"])')"
  python3 "$ROOT/scripts/mlx-agent" install opencode --scope "$scope" --project "$project" --confirm --preview-hash "$preview_hash" --json >/dev/null
}

uninstall_scope() {
  scope="$1"
  project="$2"
  plan="$(python3 "$ROOT/scripts/mlx-agent" uninstall opencode --scope "$scope" --project "$project" --dry-run --json)"
  preview_hash="$(printf '%s' "$plan" | python3 -c 'import json, sys; print(json.loads(sys.stdin.read())["data"]["preview"]["preview_hash"])')"
  python3 "$ROOT/scripts/mlx-agent" uninstall opencode --scope "$scope" --project "$project" --confirm --preview-hash "$preview_hash" --json >/dev/null
}

assert_package() {
  package_root="$1"
  config_path="$2"
  test -f "$config_path"
  test -f "$package_root/agents/mlx-advisor.md"
  for command in mlx-scout mlx-adopt mlx-wire; do
    test -f "$package_root/commands/$command.md"
    test -f "$package_root/skills/$command/SKILL.md"
  done
}

PROJECT="$SMOKE_ROOT/project"
mkdir -p "$PROJECT"
install_scope user "$PROJECT"
USER_ROOT="$HOME/.config/opencode"
assert_package "$USER_ROOT" "$USER_ROOT/opencode.json"

# `agent list` is an official local discovery surface. Command discovery has no
# documented non-model CLI, so the installed command filenames are asserted
# directly and any model-backed invocation stays explicitly opt-in below.
AGENTS="$(opencode agent list 2>&1 || true)"
case "$AGENTS" in
  *mlx-advisor*) ;;
  *) echo "OpenCode did not discover isolated mlx-advisor" >&2; exit 1 ;;
esac

# This self-contained launcher proof is separate from OpenCode discovery. The
# fixture makes no network request and does not assert an OpenCode model reply.
MLX_AGENT_FIXTURE="$ROOT/tests/fixtures/scout_responses.json" \
  python3 "$USER_ROOT/skills/mlx-scout/scripts/mlx-agent" discover --limit 1 --json >/dev/null

# A real command invocation needs a configured OpenCode provider. Never copy
# caller auth into the isolated home and never claim a model response if no auth
# exists. Users may opt in after configuring disposable credentials themselves.
if [ "${MLX_AGENT_OPENCODE_LIVE_COMMAND_DISCOVERY:-}" = "1" ]; then
  MLX_AGENT_FIXTURE="$ROOT/tests/fixtures/scout_responses.json" \
    opencode run --command mlx-scout --dir "$PROJECT" --format json -- '--limit 1 --json'
else
  echo 'SKIP: set MLX_AGENT_OPENCODE_LIVE_COMMAND_DISCOVERY=1 with disposable provider auth to invoke /mlx-scout'
fi

install_scope project "$PROJECT"
assert_package "$PROJECT/.opencode" "$PROJECT/opencode.json"
(cd "$PROJECT" && opencode agent list >/dev/null)

uninstall_scope project "$PROJECT"
test ! -e "$PROJECT/.opencode/commands/mlx-scout.md"
test ! -e "$PROJECT/opencode.json"
uninstall_scope user "$PROJECT"
test ! -e "$USER_ROOT/commands/mlx-scout.md"
test ! -e "$USER_ROOT/opencode.json"

echo 'PASS: OpenCode user and project packages installed, locally discovered, fixture-bundle-tested, and removed cleanly'
