#!/usr/bin/env bash
# run_train_lgbm_kt.sh — KT-target LGBM. See scripts/train_lgbm_kt.py for the rationale.
#
# Output:
#   submissions/lgbm_kt_v1.csv
#   submissions/lgbm_kt_v1_log.txt

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm

$PY scripts/train_lgbm_kt.py
