#!/usr/bin/env bash
# v13 RFE: one-shot feature importance on v10's feature set, drop bottom
# 30% by gain, then run normal 6-fold CV + refit + save with the lean set.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm
$PY scripts/train_lgbm_v13_rfe.py
