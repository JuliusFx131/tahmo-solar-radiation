#!/usr/bin/env bash
# v10 features + LSA-SAF MDSSFTD (dssf / fdiff / dssf_direct / qflag).
# The diffuse-fraction (`fdiff`) is the LB leader's signature "Inversion" feature.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm
$PY scripts/train_lgbm_v16_mdssftd.py
