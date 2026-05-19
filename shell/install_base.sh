#!/usr/bin/env bash
# install_base.sh — core deps for the notebooks (starter + visualization)
# Targets python3.10 because that is where Jupyter's ipykernel lives in this env.
# Run once:  bash scripts/install_base.sh

set -euo pipefail

PY="${PYTHON:-python3.10}"

echo "Using interpreter: $($PY -c 'import sys; print(sys.executable, sys.version)')"
echo

$PY -m pip install --quiet --upgrade pip
$PY -m pip install --quiet \
    "numpy" \
    "pandas" \
    "scikit-learn" \
    "matplotlib" \
    "seaborn" \
    "scipy" \
    "ipykernel"

echo
echo "Base deps installed. Verifying:"
$PY -c "import pandas, numpy, sklearn, matplotlib, seaborn, scipy; \
print(f'  pandas={pandas.__version__}  numpy={numpy.__version__}  sklearn={sklearn.__version__}')"
