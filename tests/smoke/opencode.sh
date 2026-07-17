#!/usr/bin/env bash
# Smoke native OpenCode artifacts and its no-shell custom-tool transport.
set -euo pipefail

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
mkdir -p "$HOME/.config/opencode" "$XDG_CACHE_HOME" "$XDG_DATA_HOME"
printf '{"model":"unowned-global"}\n' > "$HOME/.config/opencode/opencode.json"

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
  test -f "$package_root/agents/mlx-advisor.md"
  test -f "$package_root/plugins/mlx-agent-command.ts"
  test -f "$package_root/src/mlx_agent/command_executor.py"
  for command in mlx-scout mlx-adopt mlx-wire; do
    test -f "$package_root/commands/$command.md"
    test -f "$package_root/skills/$command/SKILL.md"
  done
}

PROJECT="$SMOKE_ROOT/project"
mkdir -p "$PROJECT"
printf '{"model":"unowned-project"}\n' > "$PROJECT/opencode.json"
install_scope user "$PROJECT"
USER_ROOT="$HOME/.config/opencode"
assert_package "$USER_ROOT"
test "$(cat "$USER_ROOT/opencode.json")" = '{"model":"unowned-global"}'

if command -v node >/dev/null 2>&1; then
  TRANSPORT="$(MLX_AGENT_FIXTURE="$ROOT/tests/fixtures/scout_responses.json" node "$ROOT/tests/fixtures/opencode_tool_transport.mjs" "$USER_ROOT/src" scout '--limit 1 --json')"
  printf '%s' "$TRANSPORT" | python3 -c 'import json, sys; value=json.load(sys.stdin); assert value["status"] == "ok" and "discover" in value["stdout"]["text"]'
else
  printf '%s' '--limit 1 --json' | MLX_AGENT_FIXTURE="$ROOT/tests/fixtures/scout_responses.json" \
    PYTHONPATH="$USER_ROOT/src" python3 -m mlx_agent.command_executor --provider opencode --capability scout >/dev/null
fi

install_scope project "$PROJECT"
assert_package "$PROJECT/.opencode"
test "$(cat "$PROJECT/opencode.json")" = '{"model":"unowned-project"}'

if command -v opencode >/dev/null 2>&1; then
  AGENTS="$(opencode agent list 2>&1 || true)"
  case "$AGENTS" in
    *mlx-advisor*) ;;
    *) echo "OpenCode did not discover isolated mlx-advisor" >&2; exit 1 ;;
  esac
  # This requires disposable credentials supplied by the caller; no model
  # response is claimed when they are absent.
  if [ "${MLX_AGENT_OPENCODE_LIVE_COMMAND_DISCOVERY:-}" = "1" ]; then
    MLX_AGENT_FIXTURE="$ROOT/tests/fixtures/scout_responses.json" \
      opencode run --command mlx-scout --dir "$PROJECT" --format json -- '--limit 1 --json'
  fi
else
  echo "SKIP: OpenCode CLI unavailable"
fi

uninstall_scope project "$PROJECT"
test ! -e "$PROJECT/.opencode/plugins/mlx-agent-command.ts"
test "$(cat "$PROJECT/opencode.json")" = '{"model":"unowned-project"}'
uninstall_scope user "$PROJECT"
test ! -e "$USER_ROOT/plugins/mlx-agent-command.ts"
test "$(cat "$USER_ROOT/opencode.json")" = '{"model":"unowned-global"}'

echo 'PASS: OpenCode package installed, custom-tool transport fixture ran, and owned artifacts were removed cleanly'
