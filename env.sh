#!/usr/bin/env bash
# =============================================================================
#  Qwen3.8-27B-w8a8  |  vLLM-Ascend  |  shared runtime environment
#  Sourced by start_server.sh / stop_server.sh / healthcheck.sh
#  All values can be overridden from the caller environment.
# =============================================================================

# ---- CANN / Ascend toolkit --------------------------------------------------
# The vendor set_env.sh scripts reference unset shell variables, so relax
# `set -u` while sourcing them.
case $- in *u*) _ENV_HAD_U=1; set +u ;; *) _ENV_HAD_U=0 ;; esac
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
[ "${_ENV_HAD_U}" = "1" ] && set -u
unset _ENV_HAD_U

# ---- Device selection -------------------------------------------------------
# Defaults to the first four visible NPUs. Point it at whichever devices you
# want to serve on, e.g. ASCEND_RT_VISIBLE_DEVICES=2,3,4,5 ./start_server.sh
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"

# ---- vLLM-Ascend runtime tuning --------------------------------------------
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
# NZ weight layout -> best MATMUL throughput on 910B
export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-1}"
# FlashComm1 (sequence-parallel). NOTE: with vLLM-Ascend 0.23.0 it is not
# compatible with the dense Qwen3.5 GDN attention path during decode-graph
# capture, so it is disabled by default. It only helps MoE/EP workloads anyway.
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-0}"
# Disable the vLLM compilation cache path that is not used on Ascend
export VLLM_ASCEND_ENABLE_MLP_OPTIMIZE="${VLLM_ASCEND_ENABLE_MLP_OPTIMIZE:-0}"

# ---- HCCL collective communication tuning -----------------------------------
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"

# ---- Host / NPU runtime tuning ---------------------------------------------
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"              # async task queue
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"             # bind worker cores
export ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-0}"

# ---- Model artefact ---------------------------------------------------------
# Either drop/symlink the weights into ./models/Qwen3.8-27B-w8a8 or point
# MODEL_PATH at wherever you downloaded them.
export MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR:-$PWD}/models/Qwen3.8-27B-w8a8}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.8-27B}"
