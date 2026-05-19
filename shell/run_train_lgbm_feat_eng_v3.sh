#!/usr/bin/env bash
# run_train_lgbm_feat_eng_v3.sh — feat_eng with the new external data:
#   • pvlib clear-sky (Ineichen-Perez + Haurwitz + Linke turbidity)
#   • Open-Meteo GHI/DNI/DHI, layered cloud cover, wind, dewpoint
#   • Per-(station, hour) night override
#
# Output:
#   submissions/lgbm_feat_eng_v3.csv
#   submissions/lgbm_feat_eng_v3_log.txt

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm

$PY scripts/train_lgbm_feat_eng_v3.py
