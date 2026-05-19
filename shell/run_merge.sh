#!/usr/bin/env bash
# run_merge.sh — Final step: merge ALL available external CSVs into Train/Test.
# Run this ONCE after every other run_*.sh has finished (or has at least produced
# its output CSV in data/satellite/). Missing sources are skipped with a warning.
# Writes data/processed/Train_enhanced.csv and Test_enhanced.csv.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

$PY -c "$PY_PREAMBLE
from data_pipeline import merge_and_save
merge_and_save()
"
