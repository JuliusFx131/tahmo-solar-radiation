#!/usr/bin/env bash
# run_train_lgbm_feat_eng_v2.sh — adds anomaly_score feature, rolls of
# tcwv/blh, and per-(station, hour) night override. See script header.
#
# Output:
#   submissions/lgbm_feat_eng_v2.csv
#   submissions/lgbm_feat_eng_v2_log.txt

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm

$PY scripts/train_lgbm_feat_eng_v2.py
