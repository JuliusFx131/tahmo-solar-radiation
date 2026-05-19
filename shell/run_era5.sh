#!/usr/bin/env bash
# run_era5.sh — Source 4/7: ERA5 hourly reanalysis via Copernicus CDS.
# Writes data/satellite/era5_hourly.csv (ext_era5_* cols) and per-month NetCDFs.
# Credentials: CDS_KEY (from _env.sh).
# Resumable: per-month NetCDFs persist; checkpoint CSV updates monthly.
# Long-running — CDS queues requests; can take many hours.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

: "${CDS_KEY:?missing in _env.sh}"

$PY -c "$PY_PREAMBLE
import os
from data_pipeline import extract_era5
extract_era5(os.environ['CDS_KEY'])
"
