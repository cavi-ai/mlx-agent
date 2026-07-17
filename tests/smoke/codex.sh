#!/usr/bin/env bash
# Smoke the public Codex plugin lifecycle without touching the caller's home.
set -euo pipefail

if ! command -v codex >/dev/null 2>&1; then
  echo "SKIP: Codex CLI unavailable"
  exit 0
fi

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
SMOKE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mlx-agent-codex-smoke.XXXXXX")"
CALLER_HOME="${HOME:?HOME must be set for the Codex smoke}"
CALLER_CODEX_HOME="${CODEX_HOME:-$CALLER_HOME/.codex}"
cleanup() {
  rm -rf "$SMOKE_ROOT"
}
trap cleanup EXIT

if [ ! -f "$CALLER_CODEX_HOME/auth.json" ]; then
  echo "ERROR: authenticated Codex login is required for end-to-end smoke" >&2
  exit 1
fi

export CODEX_HOME="$SMOKE_ROOT/codex-home"
export HOME="$SMOKE_ROOT/home"
export XDG_CONFIG_HOME="$SMOKE_ROOT/config"
export XDG_CACHE_HOME="$SMOKE_ROOT/cache"
MARKETPLACE_ROOT="$SMOKE_ROOT/marketplace"
MARKETPLACE_PATH="$MARKETPLACE_ROOT/.agents/plugins/marketplace.json"
mkdir -p "$CODEX_HOME" "$HOME" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" "$(dirname "$MARKETPLACE_PATH")" "$MARKETPLACE_ROOT/plugins"
# Keep the caller's credential private: copy only its auth file into the
# disposable Codex home, never print it, and remove the whole home on exit.
cp "$CALLER_CODEX_HOME/auth.json" "$CODEX_HOME/auth.json"
chmod 600 "$CODEX_HOME/auth.json"
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

# Ask Codex itself to render a model-visible prompt after installation. This
# checks that every plugin skill is discovered before the noninteractive Scout
# invocation below reaches the model.
PROMPT_INPUT="$(codex debug prompt-input '$mlx-agent:mlx-scout $mlx-agent:mlx-adopt $mlx-agent:mlx-wire')"
python3 -c '
import json
import sys
value = json.loads(sys.argv[1])
rendered = json.dumps(value)
for skill in ("mlx-agent:mlx-scout", "mlx-agent:mlx-adopt", "mlx-agent:mlx-wire"):
    if skill not in rendered:
        raise SystemExit("installed Codex skill is absent from prompt input: " + skill)
' "$PROMPT_INPUT"

# Codex skills are invoked by the model as `$mlx-agent:mlx-scout`,
# `$mlx-agent:mlx-adopt`, and `$mlx-agent:mlx-wire`; custom `/mlx-*` slash
# commands are unsupported. Invoke Scout through Codex's noninteractive session
# surface with the fixture-only source.
WORKSPACE="$SMOKE_ROOT/workspace"
EXECUTION_RESULT="$SMOKE_ROOT/codex-exec-result.txt"
mkdir -p "$WORKSPACE"
MLX_AGENT_FIXTURE="$ROOT/tests/fixtures/scout_responses.json" \
  codex exec --ephemeral --skip-git-repo-check --sandbox read-only \
    --add-dir "$ROOT/tests/fixtures" --cd "$WORKSPACE" \
    --output-last-message "$EXECUTION_RESULT" \
    '$mlx-agent:mlx-scout. Run the installed Scout skill with the provided fixture only, limit results to one, and return a structured JSON result naming operation discover and fixture model lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-Q8. End your response with MLX_AGENT_SMOKE_EXECUTION_MARKER. Do not read workspace files, tokens, secrets, real paths, or any data outside the provided fixture.'
python3 -c '
import pathlib
import sys
result = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
for expected in (
    "MLX_AGENT_SMOKE_EXECUTION_MARKER",
    "discover",
    "lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-Q8",
):
    if expected not in result:
        raise SystemExit("Codex Scout execution proof is missing: " + expected)
' "$EXECUTION_RESULT"

# Keep direct launcher execution as a separate self-contained bundle proof; it
# does not substitute for the Codex session invocation above.
MLX_AGENT_FIXTURE="$ROOT/tests/fixtures/scout_responses.json" \
  python3 "$MARKETPLACE_ROOT/plugins/mlx-agent/skills/mlx-scout/scripts/mlx-agent" discover --limit 1 --json >/dev/null
codex plugin remove mlx-agent --marketplace mlx-agent-smoke
echo 'PASS: Codex plugin installed, exposed $mlx-agent:mlx-scout, and removed in an isolated Codex home'
