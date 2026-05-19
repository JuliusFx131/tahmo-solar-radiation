#!/usr/bin/env bash
# run_merra2.sh — MERRA-2 hourly aerosols per station.
# Uses NASA Earthdata token (EARTHDATA_TOKEN in _env.sh).
# REQUIRES: one-time subscription to "NASA GESDISC DATA ARCHIVE" application
# at https://urs.earthdata.nasa.gov/  (otherwise 401 on downloads).
#
# Output:
#   data/satellite/merra2_aerosols.csv  (ext_m2_* columns)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

: "${EARTHDATA_TOKEN:?missing in _env.sh}"

$PY scripts/extract_merra2_aerosols.py
