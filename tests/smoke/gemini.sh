#!/usr/bin/env bash
set -euo pipefail

if ! command -v gemini >/dev/null 2>&1; then
  echo "SKIP: Gemini CLI unavailable"
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMOKE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mlx-agent-gemini-smoke.XXXXXX")"
cleanup() {
  rm -rf "$SMOKE_ROOT"
}
trap cleanup EXIT

export HOME="$SMOKE_ROOT/home"
export XDG_CONFIG_HOME="$SMOKE_ROOT/xdg-config"
export XDG_STATE_HOME="$SMOKE_ROOT/xdg-state"
mkdir -p "$HOME"

USER_PLAN="$(python3 "$ROOT/scripts/mlx-agent" install gemini --scope user --dry-run --json)"
USER_HASH="$(printf '%s' "$USER_PLAN" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["preview"]["preview_hash"])')"
python3 "$ROOT/scripts/mlx-agent" install gemini --scope user --confirm --preview-hash "$USER_HASH" --json >/dev/null
test -f "$HOME/.gemini/extensions/mlx-agent/gemini-extension.json"
test -f "$HOME/.gemini/extensions/mlx-agent/skills/mlx-scout/SKILL.md"

MLX_AGENT_FIXTURE="$ROOT/tests/fixtures/scout_responses.json" \
  python3 "$HOME/.gemini/extensions/mlx-agent/scripts/mlx-agent" discover --limit 1 --json >/dev/null

USER_REMOVE="$(python3 "$ROOT/scripts/mlx-agent" uninstall gemini --scope user --dry-run --json)"
USER_REMOVE_HASH="$(printf '%s' "$USER_REMOVE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["preview"]["preview_hash"])')"
python3 "$ROOT/scripts/mlx-agent" uninstall gemini --scope user --confirm --preview-hash "$USER_REMOVE_HASH" --json >/dev/null
test ! -e "$HOME/.gemini/extensions/mlx-agent/gemini-extension.json"

echo "Gemini CLI isolated lifecycle: passed"
