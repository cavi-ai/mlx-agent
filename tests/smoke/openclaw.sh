#!/usr/bin/env bash
# Smoke the OpenClaw provider plugin against a front-door model list without touching the caller's OpenClaw state.
set -euo pipefail

if ! command -v openclaw >/dev/null 2>&1; then
  echo "SKIP: OpenClaw CLI unavailable"
  exit 0
fi

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
SMOKE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mlx-agent-openclaw-smoke.XXXXXX")"
UPSTREAM_PID=""
cleanup() {
  status=$?
  if [ -n "$UPSTREAM_PID" ]; then
    kill "$UPSTREAM_PID" 2>/dev/null || true
    wait "$UPSTREAM_PID" 2>/dev/null || true
  fi
  rm -rf "$SMOKE_ROOT"
  exit "$status"
}
trap cleanup EXIT

unset OPENCLAW_HOME OPENCLAW_PROFILE OPENCLAW_GATEWAY_PORT OPENCLAW_GATEWAY_URL OPENCLAW_GATEWAY_TOKEN
export OPENCLAW_STATE_DIR="$SMOKE_ROOT/state"
export OPENCLAW_CONFIG_PATH="$OPENCLAW_STATE_DIR/openclaw.json"
export HOME="$SMOKE_ROOT/home"
export XDG_CONFIG_HOME="$SMOKE_ROOT/config"
export XDG_CACHE_HOME="$SMOKE_ROOT/cache"
MLX_AGENT_API_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
export MLX_AGENT_API_KEY
mkdir -p "$OPENCLAW_STATE_DIR" "$HOME" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"

python3 "$ROOT/tests/smoke/openclaw_upstream.py" "$SMOKE_ROOT/upstream.port" &
UPSTREAM_PID=$!
for _ in $(seq 50); do
  [ -s "$SMOKE_ROOT/upstream.port" ] && break
  sleep 0.1
done
UPSTREAM_PORT="$(cat "$SMOKE_ROOT/upstream.port")"
GATEWAY_PORT="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"

# The gateway port is unused: with no gateway lock in this state directory the CLI reads the local catalog.
cat > "$OPENCLAW_CONFIG_PATH" <<JSON
{
  "gateway": { "mode": "local", "port": $GATEWAY_PORT },
  "agents": { "defaults": { "modelPolicy": { "allow": ["mlx-agent/*"] } } },
  "models": {
    "providers": {
      "mlx-agent": {
        "baseUrl": "http://127.0.0.1:$UPSTREAM_PORT/v1",
        "api": "openai-completions",
        "apiKey": "\${MLX_AGENT_API_KEY}",
        "models": []
      }
    }
  }
}
JSON

openclaw plugins install --link --force --accept-capabilities "$ROOT/providers/openclaw" >"$SMOKE_ROOT/install.log" 2>&1 || {
  cat "$SMOKE_ROOT/install.log" >&2
  echo "ERROR: OpenClaw plugin install failed" >&2
  exit 1
}

openclaw models list --refresh --provider mlx-agent --json >"$SMOKE_ROOT/models.json" 2>"$SMOKE_ROOT/models.err" || {
  cat "$SMOKE_ROOT/models.json" "$SMOKE_ROOT/models.err" >&2
  echo "ERROR: OpenClaw model discovery failed" >&2
  exit 1
}

python3 - "$SMOKE_ROOT/models.json" <<'PY'
import json
import sys

keys = sorted(row["key"] for row in json.load(open(sys.argv[1]))["models"])
if keys != ["mlx-agent/smoke-chat-model"]:
    sys.exit("ERROR: expected only mlx-agent/smoke-chat-model, got {0}".format(keys))
PY
echo "PASS: OpenClaw discovered the chat route and left the embedding route out"
