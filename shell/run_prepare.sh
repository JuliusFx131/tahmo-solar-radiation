#!/usr/bin/env bash
# run_prepare.sh — build data/processed/Train_Test_Merged.csv
# (Combines Train.csv + Test.csv with a 'split' column. Used by visualization.ipynb.)
# Independent from satellite extraction; safe to run any time.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

$PY -c "$PY_PREAMBLE
from data_pipeline import prepare_merged_base
prepare_merged_base()
"
