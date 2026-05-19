#!/usr/bin/env bash
# run_solar.sh — Source 1/7: Solar geometry (computed, no API).
# Writes data/satellite/solar_features.csv (ext_sol_* columns).
# No credentials required. Fast — under a minute.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

$PY -c "$PY_PREAMBLE
from data_pipeline import compute_solar_features
compute_solar_features()
"
