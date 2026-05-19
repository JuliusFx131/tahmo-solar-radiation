#!/usr/bin/env bash
# run_pvlib.sh — pvlib clear-sky + solar geometry features (local compute).
# No API, no auth, no download. ~30 sec wall time per the 1.3M timestamps.
#
# Output: data/satellite/pvlib_features.csv  (ext_pv_* columns)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

$PY -c "import pvlib" 2>/dev/null || $PY -m pip install --quiet pvlib

$PY scripts/extract_pvlib.py
