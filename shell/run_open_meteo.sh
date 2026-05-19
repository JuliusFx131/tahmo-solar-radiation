#!/usr/bin/env bash
# run_open_meteo.sh — Open-Meteo ERA5 archive (free, no auth).
# Per-station hourly fetch of: shortwave radiation (GHI/DNI/DHI),
# layered cloud cover (low/mid/high), wind, CAPE, dewpoint, etc.
# Resumable: per-station CSVs cached under data/satellite/open_meteo_per_station/.
#
# Output:
#   data/satellite/open_meteo_hourly.csv

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

$PY scripts/extract_open_meteo.py
