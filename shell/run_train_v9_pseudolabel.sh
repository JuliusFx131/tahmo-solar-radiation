#!/usr/bin/env bash
# v9 pseudo-labeling: take high-confidence test predictions from v4+v7+v8
# (low std across models, daytime only) and add them back to training with
# weight 0.5. Then retrain v8 features on the augmented set.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
$PY -c "import lightgbm" 2>/dev/null || $PY -m pip install --quiet lightgbm
$PY scripts/train_lgbm_v9_pseudolabel.py
