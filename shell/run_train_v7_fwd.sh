#!/usr/bin/env bash
# run_train_v7_fwd.sh — v4 features + FORWARD weather lags (test rows look
# at their own future temperature / humidity / precipitation in the same
# test month). Fair game in this interpolation problem; high-information.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm
$PY scripts/train_lgbm_v7_fwd.py
