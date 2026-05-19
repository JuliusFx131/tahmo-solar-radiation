#!/usr/bin/env bash
# run_modis.sh — Source 6/7: MODIS daily cloud + aerosol via NASA LAADS DAAC.
# Writes data/satellite/modis_daily.csv (ext_modis_* cols).
# Credentials: EARTHDATA_TOKEN (from _env.sh).
# REQUIRES python-hdf4 — install_pipeline.sh may have skipped it on numpy 2.x.
# To enable: create a venv with `numpy<2`, then `pip install --no-build-isolation python-hdf4`.
# Resumable. Long-running.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

: "${EARTHDATA_TOKEN:?missing in _env.sh}"

# Pre-flight: warn cleanly if pyhdf is unavailable.
if ! $PY -c "import pyhdf" 2>/dev/null; then
    echo "ERROR: pyhdf (python-hdf4) is not installed in $PY."
    echo "MODIS extraction cannot run. See header of this script for the workaround."
    exit 1
fi

$PY -c "$PY_PREAMBLE
import os
from data_pipeline import extract_modis
extract_modis(os.environ['EARTHDATA_TOKEN'])
"
