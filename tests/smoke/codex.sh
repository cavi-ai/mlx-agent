#!/usr/bin/env bash
# Smoke the public Codex plugin lifecycle without touching the caller's home.
set -euo pipefail

if ! command -v codex >/dev/null 2>&1; then
  echo "SKIP: Codex CLI unavailable"
  exit 0
fi

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
SMOKE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mlx-agent-codex-smoke.XXXXXX")"
cleanup() {
  rm -rf "$SMOKE_ROOT"
}
trap cleanup EXIT

export CODEX_HOME="$SMOKE_ROOT/codex-home"
export HOME="$SMOKE_ROOT/home"
export XDG_CONFIG_HOME="$SMOKE_ROOT/config"
export XDG_CACHE_HOME="$SMOKE_ROOT/cache"
MARKETPLACE_ROOT="$SMOKE_ROOT/marketplace"
MARKETPLACE_PATH="$MARKETPLACE_ROOT/.agents/plugins/marketplace.json"
mkdir -p "$CODEX_HOME" "$HOME" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" "$(dirname "$MARKETPLACE_PATH")" "$MARKETPLACE_ROOT/plugins"
cp -R "$ROOT/providers/codex" "$MARKETPLACE_ROOT/plugins/mlx-agent"

python3 -c '
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "name": "mlx-agent-smoke",
    "interface": {"displayName": "MLX Agent Smoke"},
    "plugins": [{
        "name": "mlx-agent",
        "source": {"source": "local", "path": "./plugins/mlx-agent"},
        "policy": {"installation": "AVAILABLE", "authentication": "ON_INSTALL"},
        "category": "Developer Tools",
    }],
}, indent=2) + "\n", encoding="utf-8")
' "$MARKETPLACE_PATH"

codex plugin marketplace add "$MARKETPLACE_ROOT"
AVAILABLE="$(codex plugin list --marketplace mlx-agent-smoke --available --json)"
python3 -c '
import json
import sys
def contains_mlx_agent(value):
    if isinstance(value, dict):
        return value.get("name") == "mlx-agent" or any(contains_mlx_agent(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_mlx_agent(item) for item in value)
    return False
if not contains_mlx_agent(json.loads(sys.argv[1])):
    raise SystemExit("mlx-agent not listed by isolated Codex marketplace")
' "$AVAILABLE"
codex plugin add mlx-agent --marketplace mlx-agent-smoke

# Codex skills are invoked by the model as `$mlx-scout`, `$mlx-adopt`, and
# `$mlx-wire`; custom `/mlx-*` slash commands are unsupported. Prove the
# bundled Scout runtime works after the same local package installation.
MLX_AGENT_FIXTURE="$ROOT/tests/fixtures/scout_responses.json" \
  python3 "$MARKETPLACE_ROOT/plugins/mlx-agent/skills/mlx-scout/scripts/mlx-agent" discover --limit 1 --json >/dev/null
codex plugin remove mlx-agent --marketplace mlx-agent-smoke
echo 'PASS: Codex plugin installed, exposed $mlx-scout, and removed in an isolated Codex home'
