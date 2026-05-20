#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

# ── Defaults ──────────────────────────────────────────────────────────────────
RUN_ID=1
MAX_GEN=5
META_MODEL="gemini/gemini-3.1-pro-preview"
TASK_MODEL="openai/gpt-oss-120b"
BACKEND="openhands"

# ── CLI args ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_id)      RUN_ID="$2";     shift 2 ;;
    --max_gen)     MAX_GEN="$2";    shift 2 ;;
    --meta_model)  META_MODEL="$2"; shift 2 ;;
    --task_model)  TASK_MODEL="$2"; shift 2 ;;
    --backend)     BACKEND="$2";    shift 2 ;;
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
  --task_dir    ./tasks/denoising \
  --max_gen     "${MAX_GEN}" \
  --run_id      "${RUN_ID}" \
  --backend     "${BACKEND}" \
  --meta_model  "${META_MODEL}" \
  --task_model  "${TASK_MODEL}"
