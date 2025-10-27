#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PATH="$HOME/.local/bin:$PATH"

MODAL_RUN_TIMEOUT=${MODAL_RUN_TIMEOUT:-120}

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
    echo "[run] Command timed out after ${seconds}s: $*" >&2
  fi
  return $status
}

cd "${SCRIPT_DIR}"
echo "[run] Launching Modal raw evaluation..."
if ! with_timeout "${MODAL_RUN_TIMEOUT}" uv run modal run tools/modal_raw.py; then
  echo "[run] Modal launch failed; resolve the error above (likely missing token) and retry." >&2
  exit 1
fi
echo "[run] Modal job submitted. Monitor output above for remote logs."
