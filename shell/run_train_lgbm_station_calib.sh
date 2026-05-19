#!/usr/bin/env bash
# run_train_lgbm_station_calib.sh — LGBM + per-station linear calibration.
# See scripts/train_lgbm_station_calib.py header for rationale.
#
# Output:
#   submissions/lgbm_station_calib_v1.csv
#   submissions/lgbm_station_calib_v1_log.txt

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm

$PY scripts/train_lgbm_station_calib.py
