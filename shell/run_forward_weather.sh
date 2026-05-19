#!/usr/bin/env bash
# run_forward_weather.sh — compute forward (lead) lags of temperature,
# humidity, precipitation per (station, timestamp). Test rows lead from
# their own surrounding rows in the same test month — fair game in this
# interpolation problem.
#
# Output:
#   data/satellite/forward_weather.csv  (ext_fw_* columns)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
$PY scripts/extract_forward_weather.py
