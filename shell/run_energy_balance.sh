#!/usr/bin/env bash
# Energy-balance "Inversion" features — physically-motivated proxies for
# net radiation derived from temperature/humidity/BLH and forward-weather
# tendencies. dT/dt × BLH ≈ energy absorbed by the column.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
$PY scripts/extract_energy_balance.py
