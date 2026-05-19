#!/usr/bin/env bash
# Copernicus ADS — cams-solar-radiation-timeseries (15-min, per-station).
# Requires:
#   • ADS_KEY in _env.sh (same as run_cams.sh — Atmosphere Data Store auth)
#   • One-time license accept at
#     https://ads.atmosphere.copernicus.eu/datasets/cams-solar-radiation-timeseries
# Output:
#   data/satellite/cams_radiation_ts.csv  (ext_csr_* columns)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

: "${ADS_KEY:?missing in _env.sh — see shell/_env.sh for ADS_KEY setup}"

$PY scripts/extract_cams_radiation_timeseries.py
