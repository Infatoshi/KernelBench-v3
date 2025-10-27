#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash setup_and_run.sh <config-path>" >&2
  exit 1
fi

CONFIG_PATH_INPUT=$1
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${SCRIPT_DIR}
CONFIG_ABS=$(cd "${REPO_ROOT}" && realpath "${CONFIG_PATH_INPUT}")
if [ ! -f "${CONFIG_ABS}" ]; then
  echo "[setup] Config not found: ${CONFIG_ABS}" >&2
  exit 1
fi
CONFIG_REL=$(python3 - <<'PY' "${CONFIG_ABS}" "${REPO_ROOT}"
import os, sys
abs_path, repo_root = sys.argv[1], sys.argv[2]
print(os.path.relpath(abs_path, repo_root))
PY
)

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

cd "${REPO_ROOT}"
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

echo "[setup] Launching Modal raw evaluation for ${CONFIG_REL}..."
KB3_MODAL_CONFIG_PATH="${CONFIG_REL}" uv run modal run tools/modal_raw.py
