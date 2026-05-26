#!/usr/bin/env bash
# Download the paper-review benchmark datasets from OpenReview.
# Creates data/public/iclr2024.json, data/private/iclr2023.json, data/private/neurips2023.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}"

python fetch_openreview.py --task-dir "${SCRIPT_DIR}"
