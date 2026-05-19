#!/usr/bin/env bash
# v10 features (incl CAMS-radiation) + v11 MERRA-2 + pseudo-labels.
# Pseudo-label source: v10 + v11 + v8 consensus (std<12 W/m², daytime only).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm
$PY scripts/train_lgbm_v12_pseudo_v10.py
