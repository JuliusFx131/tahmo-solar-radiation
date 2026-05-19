#!/usr/bin/env bash
# install_all.sh — one-shot setup: base notebook deps + satellite-API deps.
# Run once after spinning up the environment:  bash /workspace/shell/install_all.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$HERE/install_base.sh"
echo
bash "$HERE/install_pipeline.sh"

echo
echo "All deps installed."
