#!/usr/bin/env bash
# run_tropomi.sh — Source 3/7: TROPOMI cloud + aerosol via Copernicus Data Space.
# Writes data/satellite/tropomi_cloud.csv and tropomi_aerosol.csv (ext_tro_* cols).
# Credentials: CDSE_USER, CDSE_PASSWORD (from _env.sh).
# Resumable: per-day checkpoints. Long-running.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

: "${CDSE_USER:?missing in _env.sh}"
: "${CDSE_PASSWORD:?missing in _env.sh}"

$PY -c "$PY_PREAMBLE
import os
from data_pipeline import extract_tropomi
extract_tropomi(os.environ['CDSE_USER'], os.environ['CDSE_PASSWORD'])
"
