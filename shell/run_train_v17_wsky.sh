#!/usr/bin/env bash
# v17: single LGBM on v10 features, trained with sample weights proportional
# to ext_csr_clearsky_ghi. Up-weights high-radiation rows where the LB metric
# punishes errors most.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm
$PY scripts/train_lgbm_v17_wsky.py
