#!/usr/bin/env bash
# Full v20 multi-model Optuna run: lgbm, xgb, hgb, cat — each tuned then refit
# with pseudo-labels and 6-fold CV. Sequential to stay under 8 GB cgroup.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
for pkg in xgboost catboost optuna pyarrow; do
  $PY -c "import $pkg" 2>/dev/null || $PY -m pip install --quiet $pkg
done
# regenerate features if parquets missing
[ -f data/processed/v20_train.parquet ] || $PY scripts/prepare_v20_features.py
$PY scripts/train_v20_optuna.py all
$PY scripts/build_v20_ensembles.py
