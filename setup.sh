#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PATH="$HOME/.local/bin:$PATH"

UV_TIMEOUT=${UV_TIMEOUT:-120}
MODAL_TOKEN_TIMEOUT=${MODAL_TOKEN_TIMEOUT:-120}

with_timeout() {
  local seconds=$1
  shift
  if [ "${seconds}" -eq 0 ]; then
    "$@"
    return $?
  fi
  timeout "${seconds}s" "$@"
  local status=$?
  if [ $status -eq 124 ]; then
    echo "[setup] Command timed out after ${seconds}s: $*" >&2
  fi
  return $status
}

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] Installing uv (timeout ${UV_TIMEOUT}s)..."
  with_timeout "${UV_TIMEOUT}" sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
  export PATH="$HOME/.local/bin:$PATH"
else
  echo "[setup] uv already present at $(command -v uv)"
fi

cd "${SCRIPT_DIR}"
echo "[setup] Synchronizing project dependencies with uv..."
with_timeout "${UV_TIMEOUT}" uv sync

echo "[setup] Checking Modal auth token..."
if with_timeout "${MODAL_TOKEN_TIMEOUT}" uv run python - <<'PY'
import modal.config, sys
profiles = modal.config.config_profiles()
sys.exit(0 if profiles else 1)
PY
then
  echo "[setup] Modal token already configured."
else
  echo "[setup] No Modal token detected; launching interactive 'modal token new'."
  with_timeout "${MODAL_TOKEN_TIMEOUT}" uv run modal token new
fi
