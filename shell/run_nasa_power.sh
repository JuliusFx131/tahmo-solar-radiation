#!/usr/bin/env bash
# run_nasa_power.sh — NASA POWER hourly per-station (MERRA-2 + GEOS).
# Free, no auth. Resumable per-station CSVs.
#
# Output:
#   data/satellite/nasa_power_hourly.csv  (ext_np_* columns)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

$PY scripts/extract_nasa_power.py
