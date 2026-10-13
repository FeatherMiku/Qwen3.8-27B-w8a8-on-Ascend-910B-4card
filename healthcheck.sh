#!/usr/bin/env bash
# =============================================================================
#  Poll the OpenAI /health + /v1/models endpoints until the server is ready.
#  Usage: ./healthcheck.sh [PORT] [TIMEOUT_SECONDS]
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOST="${HOST:-127.0.0.1}"
PORT="${1:-${PORT:-8000}}"
TIMEOUT="${2:-1800}"
API_KEY="${API_KEY:-}"

AUTH_HEADER=()
if [ -n "${API_KEY}" ]; then
    AUTH_HEADER=(-H "Authorization: Bearer ${API_KEY}")
fi

URL="http://${HOST}:${PORT}/health"
echo "Waiting for ${URL} (timeout ${TIMEOUT}s) ..."

start=$(date +%s)
while true; do
    code="$(curl -s -o /dev/null -w '%{http_code}' "${AUTH_HEADER[@]}" "${URL}" 2>/dev/null)"
    [ -z "${code}" ] && code="000"
    if [ "${code}" = "200" ]; then
        echo "Server is READY (health=200)."
        curl -s "${AUTH_HEADER[@]}" "http://${HOST}:${PORT}/v1/models" || true
        echo
        exit 0
    fi
    now=$(date +%s)
    if [ $((now - start)) -ge "${TIMEOUT}" ]; then
        echo "ERROR: server not ready after ${TIMEOUT}s (last http=${code})." >&2
        echo "Check the server log for the root cause." >&2
        exit 1
    fi
    sleep 5
done
