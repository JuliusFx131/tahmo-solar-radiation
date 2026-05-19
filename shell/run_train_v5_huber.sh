#!/usr/bin/env bash
# run_train_v5_huber.sh — v4 features with Huber loss (robust to outliers).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm
$PY scripts/train_lgbm_v5_huber.py
