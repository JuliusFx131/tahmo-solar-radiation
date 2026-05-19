"""
MERRA-2 hourly aerosol diagnostics (M2T1NXAER) per station.

Source: NASA GES DISC
  https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXAER.5.12.4/

Auth: NASA Earthdata bearer token (EARTHDATA_TOKEN in _env.sh).
You also need to **subscribe to the dataset** once at
  https://urs.earthdata.nasa.gov/   →  Applications → "NASA GESDISC DATA ARCHIVE"
otherwise downloads return 401.

The product is a daily NetCDF-4 file with hourly time slices on a
0.5°×0.625° global grid. We open each via OPeNDAP and pull the nearest
grid cell per station — no full-file download needed.

Variables (renamed with ext_m2_ prefix):
  TOTEXTTAU   → ext_m2_aod_total      (total AOD 550nm)
  DUEXTTAU    → ext_m2_aod_dust
  OCEXTTAU    → ext_m2_aod_oc         (organic carbon)
  BCEXTTAU    → ext_m2_aod_bc         (black carbon)
  SO4EXTTAU   → ext_m2_aod_so4
  SSEXTTAU    → ext_m2_aod_ss         (sea salt)
  TOTANGSTR   → ext_m2_angstrom       (Ångström exponent — size proxy)
  DUSMASS25   → ext_m2_pm25_dust      (PM2.5 dust surface, kg/m³)
  OCSMASS     → ext_m2_pm25_oc

Run:
  bash /workspace/shell/run_merra2.sh

Resumable: per-day-per-station extraction; cached in
  data/satellite/merra2_per_day/

Final concatenated output:
  data/satellite/merra2_aerosols.csv
"""

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT / "data" / "raw"
SAT_DIR  = ROOT / "data" / "satellite"
PER_DAY  = SAT_DIR / "merra2_per_day"
PER_DAY.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = RAW_DIR / "Train.csv"
TEST_CSV  = RAW_DIR / "Test.csv"
OUT       = SAT_DIR / "merra2_aerosols.csv"

# MERRA-2 file naming
#   MERRA2_<stream>00.tavg1_2d_aer_Nx.YYYYMMDD.nc4
# stream changes with version:
#   400 for 2010 onwards (covers all our 2016-2020 range)
BASE_URL = "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXAER.5.12.4"

VARS = ["TOTEXTTAU", "DUEXTTAU", "OCEXTTAU", "BCEXTTAU", "SO4EXTTAU", "SSEXTTAU",
        "TOTANGSTR", "DUSMASS25", "OCSMASS"]

RENAME = {
    "TOTEXTTAU": "ext_m2_aod_total",
    "DUEXTTAU":  "ext_m2_aod_dust",
    "OCEXTTAU":  "ext_m2_aod_oc",
    "BCEXTTAU":  "ext_m2_aod_bc",
    "SO4EXTTAU": "ext_m2_aod_so4",
    "SSEXTTAU":  "ext_m2_aod_ss",
    "TOTANGSTR": "ext_m2_angstrom",
    "DUSMASS25": "ext_m2_pm25_dust",
    "OCSMASS":   "ext_m2_pm25_oc",
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


def merra2_url(date: datetime) -> str:
    # Stream is 400 for files after 2010-12 (our range is 2016+)
    fname = f"MERRA2_400.tavg1_2d_aer_Nx.{date:%Y%m%d}.nc4"
    return f"{BASE_URL}/{date.year}/{date.month:02d}/{fname}"


def fetch_day(date: datetime, stations: pd.DataFrame, token: str) -> pd.DataFrame:
    """Download one daily file (in memory) and extract per-station hourly values."""
    import io
    import netCDF4 as nc4

    url = merra2_url(date)
    headers = {"Authorization": f"Bearer {token}"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, timeout=120, stream=True)
            resp.raise_for_status()
            data_bytes = resp.content
            break
        except requests.HTTPError as e:
            if e.response.status_code == 401:
                raise RuntimeError("MERRA-2 401 — make sure you've subscribed to "
                                   "'NASA GESDISC DATA ARCHIVE' at urs.earthdata.nasa.gov "
                                   "and your EARTHDATA_TOKEN is current") from e
            if attempt == MAX_RETRIES - 1:
                raise
            wait = RETRY_BACKOFF * (2 ** attempt)
            log.warning(f"  {date:%Y-%m-%d} retry {attempt + 1}/{MAX_RETRIES} "
                        f"after {wait}s: HTTP {e.response.status_code}")
            time.sleep(wait)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = RETRY_BACKOFF * (2 ** attempt)
            log.warning(f"  {date:%Y-%m-%d} retry {attempt + 1}/{MAX_RETRIES} after {wait}s: {e}")
            time.sleep(wait)

    rows = []
    with nc4.Dataset("in-memory", memory=data_bytes) as ds:
        # MERRA-2 dims: lat (361), lon (576), time (24 hourly midpoints)
        lat = ds.variables["lat"][:]
        lon = ds.variables["lon"][:]
        # times are minutes since reference date; use 'time' var
        # Build hourly timestamps as YYYY-MM-DD HH:30 (centred on hour)
        hour_ts = [datetime(date.year, date.month, date.day) + timedelta(minutes=30 + 60 * h)
                   for h in range(24)]

        # Pre-compute per-station nearest-grid indices
        for _, sta in stations.iterrows():
            ilat = int(np.argmin(np.abs(lat - sta["latitude"])))
            ilon = int(np.argmin(np.abs(lon - sta["longitude"])))
            for h, ts in enumerate(hour_ts):
                rec = {"station": sta["station"], "timestamp": ts}
                for v in VARS:
                    try:
                        val = float(ds.variables[v][h, ilat, ilon])
                        if val < -9e14 or val > 9e14:   # MERRA-2 fill
                            val = np.nan
                    except Exception:
                        val = np.nan
                    rec[RENAME[v]] = val
                rows.append(rec)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    token = os.environ.get("EARTHDATA_TOKEN", "")
    if not token:
        log.error("EARTHDATA_TOKEN env var not set — see shell/_env.sh")
        return

    stations = load_stations()
    d_start, d_end = date_range_overall()
    log.info(f"Stations: {len(stations)}, date range {d_start.date()} → {d_end.date()}")

    current = d_start
    files = []
    while current.date() <= d_end.date():
        per_path = PER_DAY / f"merra2_{current:%Y%m%d}.csv"
        if per_path.exists() and per_path.stat().st_size > 100:
            files.append(per_path)
            current += timedelta(days=1)
            continue
        try:
            df = fetch_day(current, stations, token)
            df.to_csv(per_path, index=False)
            if (current.day == 1):
                log.info(f"  {current:%Y-%m}: started new month "
                         f"(yesterday {len(df):,} rows)")
            files.append(per_path)
        except Exception as e:
            log.warning(f"  {current:%Y-%m-%d} FAILED: {e}")
        current += timedelta(days=1)

    if not files:
        log.error("No per-day files. Nothing to merge.")
        return

    log.info(f"Concatenating {len(files)} per-day files ...")
    big = pd.concat([pd.read_csv(p, parse_dates=["timestamp"]) for p in files],
                    ignore_index=True)
    big = big.drop_duplicates(subset=["station", "timestamp"]).sort_values(["station", "timestamp"])
    big.to_csv(OUT, index=False)
    log.info(f"Saved: {OUT}  ({len(big):,} rows × {len(big.columns)} cols)")
    log.info(f"Wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
