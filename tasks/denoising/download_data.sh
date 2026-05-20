#!/usr/bin/env bash
# Downloads and caches the three OpenProblems denoising benchmark datasets.
# Uses the Python loaders (not raw curl) so files are processed and cached correctly.
#
# If a dataset is already cached, prompts "Replace? [y/N]" — or use --replace to
# always overwrite without being asked.
#
# Usage:
#   bash tasks/denoising/download_data.sh [--replace] [all|pancreas|pbmc|tabula]
#   --replace  Overwrite existing cache without prompting
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export OPENPROBLEMS_CACHE_DIR="${OPENPROBLEMS_CACHE_DIR:-$HOME/.cache/openproblems}"
export PYTHONPATH="${REPO_ROOT}"

GREEN="\033[0;32m"; YELLOW="\033[1;33m"; RED="\033[0;31m"; NC="\033[0m"
log()  { printf "${GREEN}[denoising]${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}[denoising]${NC} %s\n" "$*"; }
fail() { printf "${RED}[denoising]${NC} %s\n" "$*" >&2; exit 1; }

# ── Parse flags ───────────────────────────────────────────────────────────────
REPLACE=false
_positional=()
for _arg in "$@"; do
  case "${_arg}" in
    --replace) REPLACE=true ;;
    *) _positional+=("${_arg}") ;;
  esac
done
set -- "${_positional[@]+"${_positional[@]}"}"
# ─────────────────────────────────────────────────────────────────────────────

# ── Install requirements? ─────────────────────────────────────────────────────
warn "Active Python: $(python --version 2>&1)  ($(which python))"
warn "Make sure you are in the right virtual environment before installing."
printf "${YELLOW}[denoising]${NC} Install tasks/denoising/requirements.txt? [y/N] "
read -r _answer
if [[ "${_answer}" =~ ^[Yy]$ ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv pip install -r "${SCRIPT_DIR}/requirements.txt" || pip install -r "${SCRIPT_DIR}/requirements.txt"
  else
    pip install -r "${SCRIPT_DIR}/requirements.txt"
  fi
  log "Requirements installed."
else
  log "Skipping install."
fi
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "${OPENPROBLEMS_CACHE_DIR}"

# Print the cache path for a loader call without actually loading the data.
# $1 = Python snippet ending with print(_cache_path(fn, ...))
_get_cache_path() {
  python - 2>/dev/null <<PY
import os, sys
sys.path.insert(0, '${REPO_ROOT}')
os.environ['OPENPROBLEMS_CACHE_DIR'] = '${OPENPROBLEMS_CACHE_DIR}'
from tasks.denoising._vendor.openproblems_min.data import no_cleanup
no_cleanup()
from tasks.denoising._vendor.openproblems_min.data.utils import _cache_path
$1
PY
}

_run_loader() {
  python - <<PY || fail "Loader failed — check requirements.txt is installed (pip install -r tasks/denoising/requirements.txt)"
import os, sys, ssl
sys.path.insert(0, '${REPO_ROOT}')
os.environ['OPENPROBLEMS_CACHE_DIR'] = '${OPENPROBLEMS_CACHE_DIR}'
try:
    import certifi
    ssl._create_default_https_context = lambda *a, **kw: ssl.create_default_context(*a, cafile=certifi.where(), **kw)
except ImportError:
    pass  # certifi not installed, proceed without patch
from tasks.denoising._vendor.openproblems_min.data import no_cleanup
no_cleanup()
$1
PY
}

# Check if a dataset's cache file exists; handle --replace and interactive prompt.
# Returns 0 to proceed with download, 1 to skip.
_check_cache() {
  local name="$1" cache_file="$2"
  [[ -z "${cache_file}" || ! -f "${cache_file}" ]] && return 0  # not cached → download

  if "${REPLACE}"; then
    log "  --replace: removing existing ${name} cache."
    rm -f "${cache_file}"
    return 0
  fi

  printf "${YELLOW}[denoising]${NC} ${name} is already cached. Replace? [y/N] "
  read -r _yn
  if [[ "${_yn}" =~ ^[Yy]$ ]]; then
    rm -f "${cache_file}"
    log "  Removed old cache, re-downloading ${name}..."
    return 0
  else
    log "  Skipping ${name} (already cached at ${cache_file})."
    return 1
  fi
}

download_pancreas() {
  local dest="${SCRIPT_DIR}/data/public/pancreas.h5ad"
  _check_cache "pancreas" "${dest}" || return 0

  log "Downloading pancreas dataset (inDrop1, ~13 MB processed)..."
  _run_loader "
from tasks.denoising._vendor.openproblems_min.data.pancreas import load_pancreas
adata = load_pancreas(test=False, keep_techs=['inDrop1'])
adata.write_h5ad('${dest}')
print(f'  {adata.shape[0]} cells x {adata.shape[1]} genes  →  ${dest}')
"
  log "  Pancreas ready."
}

download_pbmc() {
  local dest="${SCRIPT_DIR}/data/private/pbmc.h5ad"
  _check_cache "pbmc" "${dest}" || return 0

  log "Downloading PBMC 1k dataset (~20 MB processed)..."
  _run_loader "
from tasks.denoising._vendor.openproblems_min.data.pbmc import load_tenx_1k_pbmc
adata = load_tenx_1k_pbmc(test=False)
adata.write_h5ad('${dest}')
print(f'  {adata.shape[0]} cells x {adata.shape[1]} genes  →  ${dest}')
"
  log "  PBMC ready."
}

download_tabula() {
  local dest="${SCRIPT_DIR}/data/private/tabula.h5ad"
  _check_cache "tabula" "${dest}" || return 0

  log "Downloading Tabula Muris Senis Lung dataset (~302 MB, via CellXGene API)..."
  _run_loader "
from tasks.denoising._vendor.openproblems_min.data.tabula_muris_senis import load_tabula_muris_senis
adata = load_tabula_muris_senis(test=False, method_list=['droplet'], organ_list=['lung'])
adata.write_h5ad('${dest}')
print(f'  {adata.shape[0]} cells x {adata.shape[1]} genes  →  ${dest}')
"
  log "  Tabula ready."
}

main() {
  local target="${1:-all}"
  case "${target}" in
    all)      download_pancreas; download_pbmc; download_tabula ;;
    pancreas) download_pancreas ;;
    pbmc)     download_pbmc ;;
    tabula)   download_tabula ;;
    *) fail "Unknown target '${target}' — use: all | pancreas | pbmc | tabula" ;;
  esac
  log "All done. Data cached in: ${OPENPROBLEMS_CACHE_DIR}"
}

main "$@"
