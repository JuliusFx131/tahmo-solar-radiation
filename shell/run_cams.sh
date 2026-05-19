#!/usr/bin/env bash
# run_cams.sh — Source 5/7: CAMS aerosol optical depth via Copernicus ADS.
# Writes data/satellite/cams_aerosol.csv (ext_cams_* cols).
# Credentials: ADS_KEY + ADS_URL (NOT the CDS endpoint).
#
# PREREQUISITES (do these once before running):
#   1. Register at https://ads.atmosphere.copernicus.eu/ (CDS SSO works).
#   2. Accept the licence on the dataset page:
#        https://ads.atmosphere.copernicus.eu/datasets/cams-global-reanalysis-eac4
#   3. If your ADS PAT differs from your CDS PAT, set ADS_KEY in _env.sh.
#
# Resumable. Long-running — ADS queue.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

: "${ADS_KEY:?missing in _env.sh}"
: "${ADS_URL:?missing in _env.sh}"

$PY -c "$PY_PREAMBLE
import os
from data_pipeline import extract_cams
extract_cams(os.environ['ADS_KEY'], ads_url=os.environ['ADS_URL'])
"
