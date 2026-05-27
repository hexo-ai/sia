#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

# ── Defaults ──────────────────────────────────────────────────────────────────
RUN_ID=1
MAX_GEN=20
MAX_TURNS=50
TARGET_AGENT_TIMEOUT=1800
TASK_MODEL_TEMPERATURE=0.0
META_MODEL="gemini/gemini-3.1-pro-preview"
TASK_MODEL="openai/gpt-4o-mini"
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
printf "${YELLOW}[tau2-bench]${NC} Make sure you have downloaded the data first.\n"
printf "${YELLOW}[tau2-bench]${NC} Run 'python tasks/tau2-bench/download_data.py' if needed.\n\n"

# Export TASK_MODEL so the private evaluator can re-run the agent with the same model
export TASK_MODEL="${TASK_MODEL}"

# Write run config for private evaluator
RUN_DIR="${REPO_ROOT}/runs/run_${RUN_ID}"
mkdir -p "${RUN_DIR}"
cat > "${RUN_DIR}/run_config.json" <<EOF
{
  "task_model": "${TASK_MODEL}",
  "meta_model": "${META_MODEL}",
  "backend": "${BACKEND}",
  "max_turns": ${MAX_TURNS},
  "target_agent_timeout": ${TARGET_AGENT_TIMEOUT},
  "task_model_temperature": ${TASK_MODEL_TEMPERATURE}
}
EOF

python orchestration/orchestrator.py \
  --task_dir      ./tasks/tau2-bench \
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
