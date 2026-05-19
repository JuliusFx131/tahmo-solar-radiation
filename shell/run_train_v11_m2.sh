#!/usr/bin/env bash
# v10 features + MERRA-2 speciated aerosols (different model than CAMS).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm
$PY scripts/train_lgbm_v11_m2.py
