#!/usr/bin/env bash
# =============================================================================
#  Stop the Qwen3.8-27B-w8a8 vLLM-Ascend server (graceful, then force).
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/logs/server.pid"

kill_tree() {
    local pid="$1" sig="$2"
    # vLLM spawns worker subprocesses; signal the whole process group.
    if kill -0 "${pid}" 2>/dev/null; then
        pkill -"${sig}" -P "${pid}" 2>/dev/null || true
        kill -"${sig}" "${pid}" 2>/dev/null || true
    fi
}

PID=""
if [ -f "${PID_FILE}" ]; then
    PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
fi

if [ -z "${PID}" ] || ! kill -0 "${PID}" 2>/dev/null; then
    echo "No server pid in ${PID_FILE}; falling back to pattern match."
    PID="$(pgrep -f 'vllm serve .*Qwen3.8-27B' | head -n1 || true)"
fi

if [ -z "${PID}" ]; then
    echo "Server is not running."
    rm -f "${PID_FILE}"
    exit 0
fi

echo "Stopping server pid=${PID} ..."
kill_tree "${PID}" TERM

for _ in $(seq 1 30); do
    kill -0 "${PID}" 2>/dev/null || break
    sleep 1
done

if kill -0 "${PID}" 2>/dev/null; then
    echo "Graceful stop timed out; sending SIGKILL."
    kill_tree "${PID}" KILL
fi

rm -f "${PID_FILE}"
echo "Stopped."
