#!/usr/bin/env bash
# v7 features + same-day daily aggregates + energy-balance inversion features.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm
$PY scripts/train_lgbm_v8_inversion.py
