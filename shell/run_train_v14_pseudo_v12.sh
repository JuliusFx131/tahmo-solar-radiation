#!/usr/bin/env bash
# v14: iterated pseudo-labels. Same recipe as v12, with v12 added to the
# consensus source (anchored by v12, plus v10 + v11 for diversity).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm
$PY scripts/train_lgbm_v14_pseudo_v12.py
