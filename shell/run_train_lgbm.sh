#!/usr/bin/env bash
# run_train_lgbm.sh — train the LGBM baseline and write a Zindi submission.
#
# Prereqs:
#   1. data/processed/Train_enhanced.csv + Test_enhanced.csv  (from run_merge.sh)
#   2. data/processed/night_offset_per_station.csv  (from notebooks/visualization.ipynb Section J)
#   3. lightgbm installed (pip install lightgbm)
#
# Output:
#   submissions/lgbm_baseline_v1.csv       (Zindi format: ID,TargetMBE,TargetRMSE)
#   submissions/lgbm_baseline_v1_log.txt   (CV scores + run info, JSON)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

# Lazy-install lightgbm if missing
$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm

$PY scripts/train_lgbm_baseline.py
