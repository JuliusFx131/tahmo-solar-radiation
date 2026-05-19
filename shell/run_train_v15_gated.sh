#!/usr/bin/env bash
# v15: soft-gated 2-expert ensemble on the v10 feature set.
# Expert A (low-sun) + Expert B (high-sun), blended by sigmoid((elev-30)/10).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm
$PY scripts/train_lgbm_v15_gated.py
