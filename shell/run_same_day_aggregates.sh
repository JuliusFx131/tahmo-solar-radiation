#!/usr/bin/env bash
# Per-(station, date) daily aggregates of weather + radiation estimates.
# Propagated to every row of that day, so every test row knows its own
# day's character (max temperature, total precip, max NASA POWER GHI, etc.).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
$PY scripts/extract_same_day_aggregates.py
