#!/usr/bin/env bash
# run_train_v5_log.sh — v4 features with log1p target + bias-correction shift.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm
$PY scripts/train_lgbm_v5_log.py
