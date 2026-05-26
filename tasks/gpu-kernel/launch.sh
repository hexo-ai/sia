#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

# ── Defaults ──────────────────────────────────────────────────────────────────
RUN_ID=1
MAX_GEN=5
MAX_TURNS=70
TARGET_AGENT_TIMEOUT=1200
TASK_MODEL_TEMPERATURE=0.3
META_MODEL="gemini/gemini-3.1-pro-preview"
TASK_MODEL="tinker://55bc74de-c858-54ce-9756-e6f54d7a5a8d:train:0/sampler_weights/000049"
BACKEND="openhands"
PRIVATE_SCORES_TASK_MODELS="${TASK_MODEL}"
INCLUDE_GEN0=true

# ── CLI args ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_id)                     RUN_ID="$2";                     shift 2 ;;
    --max_gen)                    MAX_GEN="$2";                    shift 2 ;;
    --max_turns)                  MAX_TURNS="$2";                  shift 2 ;;
    --target_agent_timeout)       TARGET_AGENT_TIMEOUT="$2";       shift 2 ;;
    --task_model_temperature)     TASK_MODEL_TEMPERATURE="$2";     shift 2 ;;
    --meta_model)                 META_MODEL="$2";                 shift 2 ;;
    --task_model)                 TASK_MODEL="$2";                 shift 2 ;;
    --backend)                    BACKEND="$2";                    shift 2 ;;
    --private_scores_task_models) PRIVATE_SCORES_TASK_MODELS="$2"; shift 2 ;;
    --include_gen0)               INCLUDE_GEN0=true;               shift 1 ;;
    --no_include_gen0)            INCLUDE_GEN0=false;              shift 1 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done
# ─────────────────────────────────────────────────────────────────────────────

YELLOW="\033[1;33m"; NC="\033[0m"
printf "${YELLOW}[gpu-kernel]${NC} A CUDA GPU is required for this task.\n"
printf "${YELLOW}[gpu-kernel]${NC} Make sure torch and triton are installed in the run venv.\n\n"

python orchestration/orchestrator.py \
  --task_dir      ./tasks/gpu-kernel \
  --max_gen       "${MAX_GEN}" \
  --max_turns     "${MAX_TURNS}" \
  --target_agent_timeout "${TARGET_AGENT_TIMEOUT}" \
  --task_model_temperature "${TASK_MODEL_TEMPERATURE}" \
  --run_id        "${RUN_ID}" \
  --backend       "${BACKEND}" \
  --meta_model    "${META_MODEL}" \
  --task_model    "${TASK_MODEL}" \
  --private_scores_task_models "${PRIVATE_SCORES_TASK_MODELS}" \
  $( [[ "${INCLUDE_GEN0}" == "true" ]] && echo "--include_gen0" )
