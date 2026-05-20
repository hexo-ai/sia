#!/usr/bin/env bash
# ============================================================================
# download_task_data.sh
#
# Downloads the external datasets needed to evaluate the denoising task.
# The pancreas.h5ad file (~13 MB) is NOT bundled in the repo because it is
# fetched from figshare and cached in ~/.cache/openproblems/ at a
# hash-derived filename that the vendored openproblems loader expects.
#
# Usage:
#   bash scripts/download_task_data.sh           # download everything
#   bash scripts/download_task_data.sh denoising # only the pancreas file
#
# Idempotent: re-running skips files that already exist.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PANCREAS_URL="https://ndownloader.figshare.com/files/36086813"
PANCREAS_CACHE_DIR="${OPENPROBLEMS_CACHE_DIR:-$HOME/.cache/openproblems}"

GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m"

log()  { printf "${GREEN}[download_task_data]${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}[download_task_data]${NC} %s\n" "$*"; }
fail() { printf "${RED}[download_task_data]${NC} %s\n" "$*" >&2; exit 1; }

# Compute the exact cache filename the vendored loader will look for.
# The hash depends on the function's fully-qualified module path + args.
compute_pancreas_cache_path() {
  PYTHONPATH="${REPO_ROOT}" python - <<'PY'
from tasks.denoising._vendor.openproblems_min.data.pancreas import load_pancreas
from tasks.denoising._vendor.openproblems_min.data.utils import _cache_path
print(_cache_path(load_pancreas, test=False, keep_techs=["inDrop1"]))
PY
}

download_denoising() {
  log "denoising — pancreas.h5ad from figshare (~13 MB)"

  local pancreas_file
  pancreas_file="$(compute_pancreas_cache_path)" \
    || fail "Could not compute pancreas cache path — run: pip install -r requirements.txt"

  local basename="${pancreas_file##*/}"
  local dest="${PANCREAS_CACHE_DIR}/${basename}"

  log "  URL:  ${PANCREAS_URL}"
  log "  Dest: ${dest}"

  if [[ -f "${dest}" ]]; then
    log "  Already present (skipping). Delete to re-download."
    PANCREAS_FILE="${dest}"
    return 0
  fi

  mkdir -p "${PANCREAS_CACHE_DIR}"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --progress-bar -o "${dest}.tmp" "${PANCREAS_URL}" \
      || fail "curl failed downloading ${PANCREAS_URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget --show-progress -O "${dest}.tmp" "${PANCREAS_URL}" \
      || fail "wget failed downloading ${PANCREAS_URL}"
  else
    fail "neither curl nor wget found on PATH"
  fi

  mv "${dest}.tmp" "${dest}"
  log "  Downloaded $(du -h "${dest}" | cut -f1) → ${dest}"
  log "  Tip: export OPENPROBLEMS_CACHE_DIR=${PANCREAS_CACHE_DIR} so the evaluator finds it."
  PANCREAS_FILE="${dest}"
}

print_summary() {
  cat <<EOF

──────────────────────────────────────────────────────────────────────────────
Task data status
──────────────────────────────────────────────────────────────────────────────
  denoising : $([[ -n "${PANCREAS_FILE:-}" && -f "${PANCREAS_FILE}" ]] && echo "ready  (${PANCREAS_FILE})" || echo "NEEDS: bash scripts/download_task_data.sh denoising")
──────────────────────────────────────────────────────────────────────────────

EOF
}

main() {
  local target="${1:-all}"
  PANCREAS_FILE=""

  case "${target}" in
    all|denoising)
      download_denoising
      ;;
    -h|--help|help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      fail "unknown target '${target}' — use: all | denoising"
      ;;
  esac

  print_summary
}

main "$@"
