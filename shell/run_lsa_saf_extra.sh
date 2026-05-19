#!/usr/bin/env bash
# run_lsa_saf_extra.sh — LSA-SAF extras from IPMA's mirror.
#
# Products (NOT on EUMETSAT Data Store — only on IPMA):
#   MDSSFTD  — Downwelling Shortwave + DIFFUSE FRACTION  ← the high-value one
#   MLST     — Land Surface Temperature
#   MDSLF    — Downwelling Longwave
#
# REQUIRES (one-time):
#   1. Register at https://landsaf.ipma.pt/  (free, instant)
#   2. Uncomment + fill in shell/_env.sh:
#        export LSASAF_USER="your.email"
#        export LSASAF_PASS="your.password"
#
# Usage:
#   bash shell/run_lsa_saf_extra.sh                      # all three (slow)
#   bash shell/run_lsa_saf_extra.sh --product mdssftd    # diffuse fraction only
#   bash shell/run_lsa_saf_extra.sh --product mlst       # land surface temp
#   bash shell/run_lsa_saf_extra.sh --product mdslf      # longwave
#
# Default sampling is one timestamp per day at 12:00 UTC (configurable in the
# script's SAMPLES_PER_DAY).
#
# Output (one CSV per product):
#   data/satellite/lsa_saf_mdssftd.csv
#   data/satellite/lsa_saf_mlst.csv
#   data/satellite/lsa_saf_mdslf.csv

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"

: "${LSASAF_USER:?missing in _env.sh — register at https://landsaf.ipma.pt/ and add to _env.sh}"
: "${LSASAF_PASS:?missing in _env.sh}"

$PY scripts/extract_lsa_saf_extra.py "$@"
