#!/usr/bin/env bash
# Install model libs needed by the v20 multi-model Optuna pipeline.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
$PY -m pip install --upgrade pip
$PY -m pip install xgboost catboost optuna pyarrow
