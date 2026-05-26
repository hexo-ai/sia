#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

# ── Defaults ──────────────────────────────────────────────────────────────────
RUN_ID=1
MAX_GEN=20
MAX_TURNS=70
TARGET_AGENT_TIMEOUT=1200
TASK_MODEL_TEMPERATURE=0.3
META_MODEL="gemini/gemini-3.1-pro-preview"
#ASK_MODEL="tinker://openai/gpt-oss-120b"
TASK_MODEL="tinker://0526a884-428d-5756-8234-0d66db58a27a:train:0/sampler_weights/000005"
BACKEND="openhands"
# Default: evaluate privately with gpt-oss-120b + task_model (deduped by orchestrator)
#PRIVATE_SCORES_TASK_MODELS="tinker://0526a884-428d-5756-8234-0d66db58a27a:train:0/sampler_weights/000001,${TASK_MODEL}"
PRIVATE_SCORES_TASK_MODELS="${TASK_MODEL}"
INCLUDE_GEN0=true

# ── CLI args ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_id)                    RUN_ID="$2";                     shift 2 ;;
    --max_gen)                   MAX_GEN="$2";                    shift 2 ;;
    --max_turns)                 MAX_TURNS="$2";                  shift 2 ;;
    --target_agent_timeout)      TARGET_AGENT_TIMEOUT="$2";       shift 2 ;;
    --task_model_temperature)    TASK_MODEL_TEMPERATURE="$2";     shift 2 ;;
    --meta_model)                META_MODEL="$2";                 shift 2 ;;
    --task_model)                TASK_MODEL="$2";                 shift 2 ;;
    --backend)                   BACKEND="$2";                    shift 2 ;;
    --private_scores_task_models) PRIVATE_SCORES_TASK_MODELS="$2"; shift 2 ;;
    --include_gen0)              INCLUDE_GEN0=true;               shift 1 ;;
    --no_include_gen0)           INCLUDE_GEN0=false;              shift 1 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done
# ─────────────────────────────────────────────────────────────────────────────

# Point the vendored loader at the same cache dir used by download_data.sh
export OPENPROBLEMS_CACHE_DIR="${OPENPROBLEMS_CACHE_DIR:-$HOME/.cache/openproblems}"

YELLOW="\033[1;33m"; NC="\033[0m"
printf "${YELLOW}[denoising]${NC} Make sure you have downloaded the data and installed requirements.txt.\n"
printf "${YELLOW}[denoising]${NC} Run './download_data.sh' from this task directory if needed.\n\n"

python orchestration/orchestrator.py \
  --task_dir      ./tasks/denoising \
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
