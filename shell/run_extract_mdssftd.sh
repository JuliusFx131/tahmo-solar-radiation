#!/usr/bin/env bash
# MDSSFTD per-station 15-min time series from LSA-SAF / IPMA THREDDS.
#
# Default: full 2018-2024 range (~7 years × 365 days × 96 timesteps).
# Single-year override:
#   bash shell/run_extract_mdssftd.sh 2018
# Custom range:
#   bash shell/run_extract_mdssftd.sh --start 2018-02-01 --end 2018-02-28
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

if [[ -z "${LSASAF_USER:-}" || -z "${LSASAF_PASS:-}" ]]; then
    echo "ERROR: LSASAF_USER / LSASAF_PASS not set in shell/_env.sh" >&2
    exit 1
fi

$PY -c "import xarray, netCDF4, numpy, pandas" 2>/dev/null \
    || $PY -m pip install --quiet xarray netcdf4 h5netcdf

if [[ $# -ge 1 && "$1" =~ ^[0-9]{4}$ ]]; then
    $PY scripts/extract_mdssftd_timeseries.py --year "$1"
else
    $PY scripts/extract_mdssftd_timeseries.py "$@"
fi
