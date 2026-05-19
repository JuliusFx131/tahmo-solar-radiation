"""
LSA-SAF satellite products beyond SARAH-3 (MDSSFTD, MLST, MDSLF).

  MDSSFTD — Downwelling Shortwave Surface Flux + DIFFUSE FRACTION
  MLST    — Land Surface Temperature
  MDSLF   — Downwelling Longwave Surface Flux

All three live ONLY on LSA-SAF's IPMA mirror (NOT in the EUMETSAT Data Store).

REQUIRES:
  1. One-time registration at https://landsaf.ipma.pt/   (free)
  2. Set in shell/_env.sh:
       export LSASAF_USER="..."
       export LSASAF_PASS="..."

Data layout on the IPMA server:
  https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/<PRODUCT>/NETCDF/YYYY/MM/DD/
    NETCDF4_LSASAF_MSG_<PRODUCT>_MSG-Disk_YYYYMMDDHHMM.nc
Files are produced every 15 minutes for MDSSFTD, every 15 min for MDSLF,
and hourly for MLST. We subsample to ONE file per day at 12:00 UTC for
all three (cheaper, captures the daytime peak — adjust SAMPLES_PER_DAY
in the code if you want finer cadence).

Run:
  bash /workspace/shell/run_lsa_saf_extra.sh                    # all three
  bash /workspace/shell/run_lsa_saf_extra.sh --product mdssftd  # just one

Output (one CSV per product):
  data/satellite/lsa_saf_mdssftd.csv  (ext_lsa_dssf_total, dssf_direct_frac)
  data/satellite/lsa_saf_mlst.csv     (ext_lsa_lst)
  data/satellite/lsa_saf_mdslf.csv    (ext_lsa_dslf)
"""

import argparse
import io
import logging
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT / "data" / "raw"
SAT_DIR  = ROOT / "data" / "satellite"
SAT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = RAW_DIR / "Train.csv"
TEST_CSV  = RAW_DIR / "Test.csv"

# IPMA HTTPS host
IPMA_BASE = "https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG"

# Sample once per day to keep download volume sane.
# Each file is small (~5 MB) but 60 months × 30 days × 96 timesteps = 173k files
# would take many days. One per day = ~1800 files per product.
SAMPLES_PER_DAY = ["1200"]   # HHMM (UTC). Add e.g. "0600","1800" for more.

PRODUCTS = {
    "mdssftd": {
        "subdir":     "MDSSFTD/NETCDF",
        "name_token": "MDSSFTD",
        "variables":  ["DSSF_TOT", "DSSF_DIR_FR"],
        "col_rename": {"DSSF_TOT": "ext_lsa_dssf_total",
                       "DSSF_DIR_FR": "ext_lsa_dssf_direct_frac"},
        "out_file":   SAT_DIR / "lsa_saf_mdssftd.csv",
    },
    "mlst": {
        "subdir":     "MLST/NETCDF",
        "name_token": "MLST",
        "variables":  ["LST"],
        "col_rename": {"LST": "ext_lsa_lst"},
        "out_file":   SAT_DIR / "lsa_saf_mlst.csv",
    },
    "mdslf": {
        "subdir":     "MDSLF/NETCDF",
        "name_token": "MDSLF",
        "variables":  ["DSLF"],
        "col_rename": {"DSLF": "ext_lsa_dslf"},
        "out_file":   SAT_DIR / "lsa_saf_mdslf.csv",
    },
}

MAX_RETRIES   = 4
RETRY_BACKOFF = 5


def load_stations() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_CSV)
    return (train.groupby("station")[["latitude", "longitude"]]
            .first().reset_index())


def date_range_overall() -> tuple[datetime, datetime]:
    df_tr = pd.read_csv(TRAIN_CSV, usecols=["timestamp"])
    df_te = pd.read_csv(TEST_CSV,  usecols=["timestamp"])
    full = pd.concat([df_tr, df_te])
    full["ts"] = pd.to_datetime(full["timestamp"], format="mixed", dayfirst=True)
    return full["ts"].min().to_pydatetime(), full["ts"].max().to_pydatetime()


def url_for(product: str, date: datetime, hhmm: str) -> str:
    cfg = PRODUCTS[product]
    fname = f"NETCDF4_LSASAF_MSG_{cfg['name_token']}_MSG-Disk_{date:%Y%m%d}{hhmm}.nc"
    return f"{IPMA_BASE}/{cfg['subdir']}/{date.year}/{date.month:02d}/{date.day:02d}/{fname}"


def fetch_nc(url: str, auth: HTTPBasicAuth) -> bytes:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, auth=auth, timeout=60, stream=True)
            r.raise_for_status()
            return r.content
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 401:
                raise RuntimeError("LSA-SAF 401 — register at https://landsaf.ipma.pt/ "
                                   "and set LSASAF_USER / LSASAF_PASS in _env.sh") from e
            if status == 404:
                return b""   # file just doesn't exist for that timestamp
            if attempt == MAX_RETRIES - 1:
                raise
            wait = RETRY_BACKOFF * (2 ** attempt)
            log.warning(f"  retry {attempt+1}/{MAX_RETRIES} after {wait}s: HTTP {status}")
            time.sleep(wait)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = RETRY_BACKOFF * (2 ** attempt)
            log.warning(f"  retry {attempt+1}/{MAX_RETRIES} after {wait}s: {e}")
            time.sleep(wait)


def extract_at_stations(nc_bytes: bytes, stations: pd.DataFrame, vars_to_extract):
    """Open in-memory NetCDF (via temp file for netCDF4 backend), per-station nearest."""
    import xarray as xr
    if not nc_bytes:
        return []
    tf = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    tf.write(nc_bytes); tf.close()
    rows = []
    try:
        ds = xr.open_dataset(tf.name, engine="netcdf4")
        # MSG geostationary grid uses 'lat'/'lon' (or sometimes 'latitude'/'longitude')
        lat_name = "lat" if "lat" in ds.coords else "latitude"
        lon_name = "lon" if "lon" in ds.coords else "longitude"
        for _, sta in stations.iterrows():
            try:
                point = ds.sel({lat_name: sta["latitude"], lon_name: sta["longitude"]},
                               method="nearest", tolerance=0.5)
                rec = {"station": sta["station"]}
                for v in vars_to_extract:
                    if v in ds.data_vars:
                        val = float(point[v].values.ravel()[0])
                        if val < -9000 or val > 9e9:
                            val = np.nan
                        rec[v] = val
                    else:
                        rec[v] = np.nan
                rows.append(rec)
            except Exception:
                continue
        ds.close()
    finally:
        try: os.unlink(tf.name)
        except FileNotFoundError: pass
    return rows


def run_one_product(key: str, stations: pd.DataFrame,
                    d_start: datetime, d_end: datetime, auth: HTTPBasicAuth):
    cfg = PRODUCTS[key]
    log.info(f"=== LSA-SAF {key.upper()} (IPMA) — sampling at {SAMPLES_PER_DAY} UTC ===")

    checkpoint = SAT_DIR / f"lsa_saf_{key}_checkpoint.csv"
    if checkpoint.exists() and checkpoint.stat().st_size > 0:
        try:
            records = pd.read_csv(checkpoint).to_dict("records")
            log.info(f"  resumed: {len(records):,} records cached")
        except Exception:
            records = []
    else:
        records = []

    current = d_start
    while current.date() <= d_end.date():
        for hhmm in SAMPLES_PER_DAY:
            url = url_for(key, current, hhmm)
            try:
                nc_bytes = fetch_nc(url, auth)
            except RuntimeError as e:
                log.error(str(e))
                return
            except Exception as e:
                log.debug(f"  {current:%Y-%m-%d} {hhmm}: {e}")
                continue
            rows = extract_at_stations(nc_bytes, stations, cfg["variables"])
            for r in rows:
                rec = {"station": r["station"], "date": current.date()}
                for src_name, dst_name in cfg["col_rename"].items():
                    rec[dst_name] = r.get(src_name, np.nan)
                records.append(rec)
        current += timedelta(days=1)
        if current.day == 1:
            log.info(f"  {key} completed {(current - timedelta(days=1)):%Y-%m}  "
                     f"(records so far: {len(records):,})")
            if records:
                pd.DataFrame(records).to_csv(checkpoint, index=False)

    if not records:
        log.error(f"No {key} data extracted.")
        return

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["station", "date"]).sort_values(["station", "date"])
    df.to_csv(cfg["out_file"], index=False)
    log.info(f"Saved: {cfg['out_file']}  ({len(df):,} rows)")
    if checkpoint.exists():
        checkpoint.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", choices=list(PRODUCTS.keys()) + ["all"], default="all")
    args = ap.parse_args()

    user = os.environ.get("LSASAF_USER", "")
    pwd  = os.environ.get("LSASAF_PASS", "")
    if not user or not pwd:
        log.error("LSASAF_USER / LSASAF_PASS missing in _env.sh.")
        log.error("Register at https://landsaf.ipma.pt/ then add to _env.sh:")
        log.error('  export LSASAF_USER="your.email"')
        log.error('  export LSASAF_PASS="your.password"')
        return

    auth = HTTPBasicAuth(user, pwd)
    stations = load_stations()
    d_start, d_end = date_range_overall()
    log.info(f"Stations: {len(stations)}, date range {d_start.date()} → {d_end.date()}")

    keys = list(PRODUCTS.keys()) if args.product == "all" else [args.product]
    for key in keys:
        run_one_product(key, stations, d_start, d_end, auth)


if __name__ == "__main__":
    main()
