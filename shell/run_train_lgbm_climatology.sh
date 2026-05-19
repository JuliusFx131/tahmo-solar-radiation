#!/usr/bin/env bash
# run_train_lgbm_climatology.sh — feat_eng + (station, hour) climatology feature.
# The climatology is refit inside each CV fold to avoid leakage.
#
# Output:
#   submissions/lgbm_climatology_v1.csv
#   submissions/lgbm_climatology_v1_log.txt

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm

$PY scripts/train_lgbm_climatology.py
