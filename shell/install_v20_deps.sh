#!/usr/bin/env bash
# Install model libs needed by the v20 multi-model Optuna pipeline.
# No credentials needed — safe to run on a fresh clone.
set -euo pipefail
PY="${PY:-python3.10}"
$PY -m pip install --quiet --upgrade pip
$PY -m pip install --quiet lightgbm xgboost catboost optuna pyarrow
