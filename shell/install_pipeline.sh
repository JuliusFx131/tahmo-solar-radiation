#!/usr/bin/env bash
# install_pipeline.sh — satellite-API deps required by scripts/data_pipeline.py.
# Adds on top of install_base.sh (it does NOT reinstall pandas/numpy/etc.).
# Run once:  bash /workspace/shell/install_pipeline.sh

set -euo pipefail

PY="${PYTHON:-python3.10}"

echo "Using interpreter: $($PY -c 'import sys; print(sys.executable, sys.version)')"
echo

# OS deps for pyhdf (HDF4) — needed for MODIS. Skip silently if apt-get is unavailable.
if command -v apt-get >/dev/null 2>&1; then
    if ! dpkg -s libhdf4-dev >/dev/null 2>&1; then
        echo "Installing libhdf4-dev (system) for pyhdf..."
        apt-get update -qq
        apt-get install -y -qq libhdf4-dev
    fi
fi

$PY -m pip install --quiet --upgrade pip

# Install everything except python-hdf4 first
$PY -m pip install --quiet \
    "eumdac" \
    "cdsapi" \
    "netCDF4" \
    "xarray" \
    "h5py" \
    "requests" \
    "tqdm" \
    "pvlib" \
    # "python-hdf4"

# python-hdf4 is ONLY needed for the MODIS source. It builds from source against
# the system libhdf4 + numpy headers and is fragile (no wheels on PyPI; breaks on
# numpy 2.x with current sdist). We attempt it but do not fail the whole install.
$PY -m pip install --quiet "setuptools" "wheel"
if $PY -m pip install --quiet --no-build-isolation "python-hdf4" 2>/tmp/pyhdf-err.log; then
    echo "  python-hdf4 installed (MODIS source available)."
else
    echo "  WARNING: python-hdf4 build failed — MODIS source will be unavailable."
    echo "  All other satellite sources (solar, lsa_saf, tropomi, era5, cams) still work."
    echo "  See /tmp/pyhdf-err.log for details. To retry: pin numpy<2 in a separate venv."
fi

echo
echo "Pipeline deps installed. Verifying:"
$PY -c "import eumdac, cdsapi, netCDF4, xarray, h5py, requests, tqdm, pvlib; \
print('  core satellite-API deps importable')"
$PY -c "import pyhdf; print('  pyhdf OK (MODIS available)')" 2>/dev/null \
    || echo "  pyhdf NOT installed — MODIS source disabled (other sources still work)"
