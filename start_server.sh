#!/usr/bin/env bash
# =============================================================================
#  Qwen3.8-27B-w8a8  |  OpenAI-compatible inference server (vLLM-Ascend)
# -----------------------------------------------------------------------------
#  Launches `vllm serve` with the configuration tuned in this repository:
#    * tensor parallel across N Ascend NPUs
#    * ModelSlim W8A8 quantization
#    * MTP speculative decoding (num_speculative_tokens=2)
#    * AllReduce + RMSNorm fusion
#    * ACL-graph capture sizes aligned to the 3 x batch decode shapes
#    * OpenAI Chat Completions + tool calling + multimodal + reasoning
#
#  Usage:
#     ./start_server.sh                        # daemonise (logs/ + pidfile)
#     FOREGROUND=1 ./start_server.sh           # run attached (Ctrl-C stops)
#     PORT=9000 MAX_MODEL_LEN=131072 ./start_server.sh
#
#  Every tunable below can be overridden from the environment.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

# ---- Service tunables -------------------------------------------------------
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"                        # NPUs used for tensor parallelism
DP_SIZE="${DP_SIZE:-1}"                        # data-parallel replicas
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"       # model native max_position_embeddings
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
QUANT="${QUANT:-ascend}"                       # ModelSlim W8A8
API_KEY="${API_KEY:-}"                         # empty -> auth disabled
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-2}"        # MTP draft length, 0 disables MTP
CG_CAPTURE_SIZES="${CG_CAPTURE_SIZES:-[1,2,3,4,6,8,12,16,24,32,48,64,72,96,128]}"
# Ascend compile-time fusions. fuse_allreduce_rms folds the TP all-reduce into
# the following residual-add + RMSNorm, roughly halving the number of HCCL
# collectives per decode step. Set to "" to disable.
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-{\"ascend_compilation_config\":{\"fuse_allreduce_rms\":true}}}"
# Report prefix-cache hits back to clients as
# usage.prompt_tokens_details.cached_tokens. vLLM defaults this to off, which is
# why clients (agent UIs) cannot show a cache-hit indicator without it.
PROMPT_TOKENS_DETAILS="${PROMPT_TOKENS_DETAILS:-1}"
# SECURITY: letting the API read arbitrary files off the server filesystem is a
# real risk, so it is disabled by default. Opt in explicitly, e.g.
#   ALLOWED_LOCAL_MEDIA_PATH=/path/to/media ./start_server.sh
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-}"
FOREGROUND="${FOREGROUND:-0}"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/server.log"
PID_FILE="${LOG_DIR}/server.pid"

# ---- Assemble the vLLM command line ----------------------------------------
SERVE_ARGS=(
    "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${HOST}"
    --port "${PORT}"
    --tensor-parallel-size "${TP_SIZE}"
    --data-parallel-size "${DP_SIZE}"
    --distributed-executor-backend mp
    --quantization "${QUANT}"
    --trust-remote-code
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    # ---- Prefix caching ----
    --enable-prefix-caching
    # ---- OpenAI tool calling (Qwen XML <tool_call> format) ----
    --enable-auto-tool-choice
    --tool-call-parser qwen3_xml
    --reasoning-parser qwen3
    # ---- Multimodal (image / video) ----
    --limit-mm-per-prompt '{"image":8,"video":2}'
    # ---- Graph capture (decode shapes aligned to MTP-2's 3 x batch tokens) ----
    --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":${CG_CAPTURE_SIZES}}"
    --seed 0
)

# Async scheduling overlaps output processing with the next forward pass.
SERVE_ARGS+=(--async-scheduling)

if [ "${SPEC_NUM_TOKENS}" != "0" ]; then
    SERVE_ARGS+=(--speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":${SPEC_NUM_TOKENS}}")
fi

if [ -n "${API_KEY}" ]; then
    SERVE_ARGS+=(--api-key "${API_KEY}")
fi

if [ -n "${ADDITIONAL_CONFIG}" ]; then
    SERVE_ARGS+=(--additional-config "${ADDITIONAL_CONFIG}")
fi

if [ "${PROMPT_TOKENS_DETAILS}" = "1" ]; then
    SERVE_ARGS+=(--enable-prompt-tokens-details)
fi

if [ -n "${ALLOWED_LOCAL_MEDIA_PATH}" ]; then
    SERVE_ARGS+=(--allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}")
fi

# ---- Banner -----------------------------------------------------------------
echo "=============================================================="
echo " Qwen3.8-27B-w8a8  vLLM-Ascend server"
echo "--------------------------------------------------------------"
echo " NPU devices      : ${ASCEND_RT_VISIBLE_DEVICES}"
echo " Tensor parallel  : ${TP_SIZE}   (data parallel: ${DP_SIZE})"
echo " Quantization     : ${QUANT}"
echo " Max model len    : ${MAX_MODEL_LEN}"
echo " MTP spec tokens  : ${SPEC_NUM_TOKENS}"
echo " Prefix cache     : on (cached_tokens reported: ${PROMPT_TOKENS_DETAILS})"
echo " Endpoint         : http://${HOST}:${PORT}/v1"
echo " Model name       : ${SERVED_MODEL_NAME}"
echo " Log              : ${LOG_FILE}"
echo "=============================================================="

# ---- Launch -----------------------------------------------------------------
if [ "${FOREGROUND}" = "1" ]; then
    exec vllm serve "${SERVE_ARGS[@]}"
else
    # keep the previous run's log for post-mortems
    if [ -f "${LOG_FILE}" ]; then
        mv -f "${LOG_FILE}" "${LOG_FILE}.1"
    fi
    nohup vllm serve "${SERVE_ARGS[@]}" >> "${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}"
    echo "Server started in background (pid $(cat "${PID_FILE}"))."
    echo "Tail logs: tail -f ${LOG_FILE}"
fi
