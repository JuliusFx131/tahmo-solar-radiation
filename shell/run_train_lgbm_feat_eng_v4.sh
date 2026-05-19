#!/usr/bin/env bash
# run_train_lgbm_feat_eng_v4.sh — feat_eng with the full external-data set:
#   • pvlib clear-sky
#   • Open-Meteo (GHI/DNI/DHI, layered clouds, wind, dewpoint)
#   • NASA POWER (MERRA-2 + GEOS — independent radiation estimate)
#   • Extended CAMS (7 species + TCWV)
#   • Per-(station, hour) night override
#
# Output:
#   submissions/lgbm_feat_eng_v4.csv
#   submissions/lgbm_feat_eng_v4_log.txt

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm

$PY scripts/train_lgbm_feat_eng_v4.py
