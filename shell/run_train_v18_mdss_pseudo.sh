#!/usr/bin/env bash
# v16 features + MDSSFTD-derived (kt, anomalies, fdiff×csr, rolling) + pseudo-labels
# from consensus of v10 / v12_pseudo / v16. m2_aod_* dropped (proved net-negative).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm
$PY scripts/train_lgbm_v18_mdss_pseudo.py
