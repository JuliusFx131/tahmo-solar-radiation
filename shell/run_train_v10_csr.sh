#!/usr/bin/env bash
# v8 features + CAMS Solar Radiation Timeseries (pre-computed 15-min
# GHI/BHI/DHI/BNI at exact station coords, all-sky + clear-sky + reliability).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm
$PY scripts/train_lgbm_v10_csr.py
