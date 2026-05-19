#!/usr/bin/env bash
# run_train_per_station.sh — one LGBM per station using v4 features
# (no temporal_neighbors). Holdout-based early stopping per station,
# then refit on full station data, predict that station's test rows.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm
$PY scripts/train_lgbm_per_station.py
