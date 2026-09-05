#!/usr/bin/env bash
set -euo pipefail

if ! command -v agy >/dev/null 2>&1; then
  echo "SKIP: Agy unavailable"
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMOKE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mlx-agent-agy-smoke.XXXXXX")"
cleanup() {
  HOME="$SMOKE_ROOT/home" agy plugin uninstall mlx-agent >/dev/null 2>&1 || true
  rm -rf "$SMOKE_ROOT"
}
trap cleanup EXIT

export HOME="$SMOKE_ROOT/home"
export XDG_CONFIG_HOME="$SMOKE_ROOT/xdg-config"
export XDG_STATE_HOME="$SMOKE_ROOT/xdg-state"
mkdir -p "$HOME"

agy plugin validate "$ROOT/providers/agy"
agy plugin install "$ROOT/providers/agy"
AGY_PLUGINS="$(agy plugin list 2>&1)"
printf '%s\n' "$AGY_PLUGINS" | grep -F "mlx-agent" >/dev/null

PROJECT="$SMOKE_ROOT/project"
mkdir -p "$PROJECT"
PROJECT_PLAN="$(python3 "$ROOT/scripts/mlx-agent" install agy --scope project --project "$PROJECT" --dry-run --json)"
PROJECT_HASH="$(printf '%s' "$PROJECT_PLAN" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["preview"]["preview_hash"])')"
python3 "$ROOT/scripts/mlx-agent" install agy --scope project --project "$PROJECT" --confirm --preview-hash "$PROJECT_HASH" --json >/dev/null
test -f "$PROJECT/.agents/plugins/mlx-agent/plugin.json"

MLX_AGENT_FIXTURE="$ROOT/tests/fixtures/scout_responses.json" \
  python3 "$PROJECT/.agents/plugins/mlx-agent/skills/mlx-scout/scripts/mlx-agent" discover --limit 1 --json >/dev/null

PROJECT_REMOVE="$(python3 "$ROOT/scripts/mlx-agent" uninstall agy --scope project --project "$PROJECT" --dry-run --json)"
PROJECT_REMOVE_HASH="$(printf '%s' "$PROJECT_REMOVE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["preview"]["preview_hash"])')"
python3 "$ROOT/scripts/mlx-agent" uninstall agy --scope project --project "$PROJECT" --confirm --preview-hash "$PROJECT_REMOVE_HASH" --json >/dev/null
test ! -e "$PROJECT/.agents/plugins/mlx-agent/plugin.json"

agy plugin uninstall mlx-agent
REMOVED="$(agy plugin list 2>&1)"
if printf '%s\n' "$REMOVED" | grep -F "mlx-agent" >/dev/null; then
  echo "Agy plugin uninstall left mlx-agent registered" >&2
  exit 1
fi

echo "Agy isolated lifecycle: passed"
