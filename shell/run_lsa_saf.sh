#!/usr/bin/env bash
# run_lsa_saf.sh — Source 2/7: SARAH-3 surface radiation via EUMETSAT.
# Writes data/satellite/sarah_radiation.csv (ext_lsa_* columns).
# Credentials: EUMETSAT_KEY, EUMETSAT_SECRET (from _env.sh).
# Resumable: checkpoint at data/satellite/sarah_checkpoint.csv.
# Long-running — many hours.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

: "${EUMETSAT_KEY:?missing in _env.sh}"
: "${EUMETSAT_SECRET:?missing in _env.sh}"

$PY -c "$PY_PREAMBLE
import os
from data_pipeline import extract_lsa_saf
extract_lsa_saf(os.environ['EUMETSAT_KEY'], os.environ['EUMETSAT_SECRET'])
"
