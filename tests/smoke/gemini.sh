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

# Gemini's extension validation above checks all native command TOML files;
# this fresh discovery command proves the installed extension exposes its
# bundled runtime skills without requiring an authenticated model session.
SKILLS="$(gemini skills list --all 2>&1)"
for capability in mlx-scout mlx-adopt mlx-wire; do
  case "$SKILLS" in
    *"$capability"*) ;;
    *) echo "Gemini did not discover bundled skill: $capability" >&2; exit 1 ;;
  esac
done

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

echo 'PASS: Gemini extension installed, validated native commands, discovered bundled skills, and removed in an isolated home'
