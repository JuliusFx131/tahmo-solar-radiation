#!/usr/bin/env bash
# run_train_v6_tneigh.sh — v4 features + temporal-neighbour radiation features.
# Requires:
#   data/satellite/temporal_neighbors.csv  (from run_temporal_neighbors.sh)
#   data/processed/Train_enhanced.csv / Test_enhanced.csv  (from run_merge.sh)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm
$PY scripts/train_lgbm_v6_tneigh.py
