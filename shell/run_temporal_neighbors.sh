#!/usr/bin/env bash
# run_temporal_neighbors.sh — build per-(station, hour, doy) rolling-mean
# radiation features from training data (precomputed once, joined onto
# the full train+test timeline at merge time).
#
# Output:
#   data/satellite/temporal_neighbors.csv  (ext_tn_* columns)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
$PY scripts/extract_temporal_neighbors.py
