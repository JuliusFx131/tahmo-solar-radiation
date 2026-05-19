"""
LSA-SAF MDSSFTD per-station 15-min time series.

MDSSFTD = MSG Downwelling Surface Shortwave Flux + diffuse fraction.

  DSSF_TOT          (W/m²) — total downwelling shortwave (same physical
                              quantity as the competition target)
  FRACTION_DIFFUSE  (—)    — diffuse / total share (the LB leader's
                              signature "Inversion" feature)
  quality_flag      (—)    — pixel QC

Why this product: the LB leader's published "Inversion" approach uses
LSA-SAF diffuse fraction. CAMS already provided clear-sky GHI; what we
were missing was a satellite-observed (not modelled) all-sky DSSF AND its
diffuse partition. MDSSFTD provides both at the exact MSG 15-min cadence
that the competition data was sampled on.

Access:
  • Source:   IPMA HTTPS (https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/MDSSFTD/NETCDF/...)
  • Format:   regular-grid lat/lon NetCDF4 (3201×3201, 0.05°, [-80,80] both)
  • Auth:     Basic auth with LSASAF_USER / LSASAF_PASS
  • Strategy: download each ~680 KB file via requests (explicit timeouts),
              open locally, slice 40 pixels via .isel(), delete temp file.
              (THREDDS OpenDAP hangs unpredictably under concurrent load;
              NCSS endpoint returns 404. Plain HTTPS + local parse is the
              only reliable path here.)

Output:
  data/satellite/mdssftd_per_station/<station>.csv
    columns: timestamp, ext_mdssftd_dssf, ext_mdssftd_fdiff,
             ext_mdssftd_dssf_direct, ext_mdssftd_qflag

Run:
  bash /workspace/shell/run_extract_mdssftd.sh                # full 2018-2024
  bash /workspace/shell/run_extract_mdssftd.sh 2018           # single year
"""

import argparse
import gc
import logging
import os
import sys
import tempfile
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT / "data" / "raw"
SAT_DIR  = ROOT / "data" / "satellite"
PER_STA  = SAT_DIR / "mdssftd_per_station"
PER_STA.mkdir(parents=True, exist_ok=True)
LOG_DIR  = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = RAW_DIR / "Train.csv"
TEST_CSV  = RAW_DIR / "Test.csv"

DATA_HOST    = "datalsasaf.lsasvcs.ipma.pt"
PRODUCT_PATH = "MSG/MDSSFTD/NETCDF"
FILENAME_FMT = "NETCDF4_LSASAF_MSG_MDSSFTD_MSG-Disk_{ymdhm}.nc"
TIME_STEP_MIN  = 15
N_WORKERS      = 8           # parallel HTTPS file downloads
CONNECT_TIMEOUT = 15         # seconds for TCP+TLS handshake
READ_TIMEOUT    = 45         # seconds to receive the ~680 KB body
MAX_RETRIES    = 3
RETRY_BACKOFF  = 4

# netCDF4's C library is NOT thread-safe → serialize all xarray open+load calls.
NC_LOCK = threading.Lock()


def https_url(dt: datetime) -> str:
    ymdhm = dt.strftime("%Y%m%d%H%M")
    return (
        f"https://{DATA_HOST}/PRODUCTS/{PRODUCT_PATH}/"
        f"{dt:%Y/%m/%d}/{FILENAME_FMT.format(ymdhm=ymdhm)}"
    )


def load_stations() -> pd.DataFrame:
    return (pd.read_csv(TRAIN_CSV, usecols=["station", "latitude", "longitude"])
            .drop_duplicates("station")
            .sort_values("station")
            .reset_index(drop=True))


def build_pixel_indices(auth: HTTPBasicAuth) -> pd.DataFrame:
    """Download one reference file, snap each station to nearest (lat_idx, lon_idx).

    The MDSSFTD grid is regular: lat ∈ [-80, 80] step -0.05°, lon ∈ [-80, 80] step 0.05°.
    Same grid across all files → compute once.
    """
    import xarray as xr
    stations = load_stations()
    ref_dt = datetime(2018, 1, 15, 12, 0)
    log.info("Downloading reference file to build (lat_idx, lon_idx) per station ...")
    nc_bytes = _http_get(https_url(ref_dt), auth)
    if not nc_bytes:
        raise RuntimeError("Could not fetch reference MDSSFTD file — auth or network?")
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tf:
        tf.write(nc_bytes); ref_path = tf.name
    try:
        ds = xr.open_dataset(ref_path, engine="netcdf4")
        lat = ds["lat"].values; lon = ds["lon"].values
        ds.close()
    finally:
        os.unlink(ref_path)
    lat_idx = np.array([np.abs(lat - la).argmin() for la in stations["latitude"]])
    lon_idx = np.array([np.abs(lon - lo).argmin() for lo in stations["longitude"]])
    stations["lat_idx"] = lat_idx
    stations["lon_idx"] = lon_idx
    stations["lat_pixel"] = lat[lat_idx]
    stations["lon_pixel"] = lon[lon_idx]
    stations["pixel_dist_deg"] = np.hypot(
        stations["latitude"] - stations["lat_pixel"],
        stations["longitude"] - stations["lon_pixel"])
    log.info(f"  max station→pixel offset: {stations['pixel_dist_deg'].max():.3f}° "
             f"(~{stations['pixel_dist_deg'].max()*111:.1f} km)")
    return stations


def _http_get(url: str, auth: HTTPBasicAuth) -> bytes:
    """Download URL with explicit (connect, read) timeouts and retry/backoff.
    Returns bytes on 200, empty bytes on 404 (file genuinely missing),
    raises on auth failure or repeated network errors.
    """
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, auth=auth,
                             timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), stream=False)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return b""
            if r.status_code == 401:
                raise RuntimeError(f"LSA-SAF 401 — check LSASAF_USER / LSASAF_PASS")
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                log.debug(f"  giving up on {url.rsplit('/',1)[-1]}: {e}")
                return b""
            time.sleep(RETRY_BACKOFF * (2 ** attempt))
    return b""


def fetch_one_timestep(dt: datetime, stations: pd.DataFrame,
                       auth: HTTPBasicAuth) -> list[dict]:
    """Download one MDSSFTD file, slice 40 station pixels, delete temp."""
    import xarray as xr
    nc_bytes = _http_get(https_url(dt), auth)
    if not nc_bytes:
        return []
    tf = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    try:
        tf.write(nc_bytes); tf.close()
        with NC_LOCK:                      # netCDF4 C lib not thread-safe
            ds = xr.open_dataset(tf.name, engine="netcdf4")
            sub = ds.isel(
                lat=xr.DataArray(stations["lat_idx"].values, dims="station"),
                lon=xr.DataArray(stations["lon_idx"].values, dims="station"),
            ).squeeze("time", drop=True).load()
            ds.close()
    except Exception as e:
        log.debug(f"  {dt:%Y-%m-%d %H:%M}: parse failed ({e})")
        return []
    finally:
        try: os.unlink(tf.name)
        except FileNotFoundError: pass

    dssf = sub["DSSF_TOT"].values.astype(np.float32)
    fdif = sub["FRACTION_DIFFUSE"].values.astype(np.float32)
    qflg = sub["quality_flag"].values.astype(np.float32)
    dssf = np.where((dssf < -1000) | (dssf > 1500), np.nan, dssf)
    fdif = np.where((fdif < 0) | (fdif > 1.5),     np.nan, fdif)

    rows = []
    for i, st in enumerate(stations["station"].values):
        d = float(dssf[i]) if np.isfinite(dssf[i]) else np.nan
        f = float(fdif[i]) if np.isfinite(fdif[i]) else np.nan
        q = float(qflg[i]) if np.isfinite(qflg[i]) else np.nan
        ddir = d * (1.0 - f) if (np.isfinite(d) and np.isfinite(f)) else np.nan
        rows.append({
            "station":               st,
            "timestamp":             dt,
            "ext_mdssftd_dssf":       d,
            "ext_mdssftd_fdiff":      f,
            "ext_mdssftd_dssf_direct": ddir,
            "ext_mdssftd_qflag":      q,
        })
    return rows


def already_done_through(station: str) -> datetime | None:
    """Return the max timestamp already cached for `station`, or None."""
    path = PER_STA / f"{station}.csv"
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path, usecols=["timestamp"])
        if df.empty:
            return None
        return pd.to_datetime(df["timestamp"]).max().to_pydatetime()
    except Exception:
        return None


def append_rows_per_station(rows: list[dict]):
    """Append fetched rows to per-station CSVs (one append per station)."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for sta, sub in df.groupby("station"):
        path = PER_STA / f"{sta}.csv"
        sub2 = sub.drop(columns=["station"]).sort_values("timestamp")
        # Atomic append: read-merge-write to .tmp, rename.
        if path.exists() and path.stat().st_size > 0:
            try:
                prev = pd.read_csv(path, parse_dates=["timestamp"])
                merged = pd.concat([prev, sub2], ignore_index=True)
                merged = merged.drop_duplicates("timestamp").sort_values("timestamp")
            except Exception:
                merged = sub2
        else:
            merged = sub2
        tmp = path.with_suffix(".csv.tmp")
        merged.to_csv(tmp, index=False)
        tmp.replace(path)


def day_timesteps(date: datetime) -> list[datetime]:
    start = datetime(date.year, date.month, date.day, 0, 0)
    return [start + timedelta(minutes=TIME_STEP_MIN * i)
            for i in range(24 * 60 // TIME_STEP_MIN)]


def extract_range(start: datetime, end: datetime,
                  stations: pd.DataFrame, auth: HTTPBasicAuth):
    """Walk day-by-day; one ThreadPool batch per day."""
    cur = start
    days_total = (end.date() - start.date()).days + 1
    days_done = 0
    while cur.date() <= end.date():
        t0 = time.time()
        timesteps = day_timesteps(cur)
        rows_today = []
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(fetch_one_timestep, ts, stations, auth): ts
                    for ts in timesteps}
            for fut in as_completed(futs):
                try:
                    rows_today.extend(fut.result())
                except Exception as e:
                    log.debug(f"  worker error: {e}")
        append_rows_per_station(rows_today)
        days_done += 1
        n_ts = len({r["timestamp"] for r in rows_today})
        log.info(f"  {cur:%Y-%m-%d}: {n_ts}/96 timesteps × {len(stations)} pixels "
                 f"= {len(rows_today):,} rows  ({time.time()-t0:.1f}s)  "
                 f"[{days_done}/{days_total} days]")
        sys.stdout.flush()
        cur += timedelta(days=1)
        gc.collect()


def date_range_from_data() -> tuple[datetime, datetime]:
    df_tr = pd.read_csv(TRAIN_CSV, usecols=["timestamp"])
    df_te = pd.read_csv(TEST_CSV,  usecols=["timestamp"])
    full = pd.concat([df_tr, df_te])
    full["ts"] = pd.to_datetime(full["timestamp"], format="mixed", dayfirst=True)
    return (datetime(full["ts"].min().year, 1, 1),
            datetime(full["ts"].max().year, 12, 31))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year",  type=int, default=None,
                    help="Restrict to a single year (default: full 2018-2024 range from data)")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD start (overrides --year)")
    ap.add_argument("--end",   default=None, help="YYYY-MM-DD end (overrides --year)")
    args = ap.parse_args()

    user = os.environ.get("LSASAF_USER", "")
    pwd  = os.environ.get("LSASAF_PASS", "")
    if not user or not pwd:
        log.error("LSASAF_USER / LSASAF_PASS missing. Source shell/_env.sh first.")
        sys.exit(1)
    auth = HTTPBasicAuth(user, pwd)

    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end   = datetime.strptime(args.end,   "%Y-%m-%d")
    elif args.year:
        start = datetime(args.year, 1, 1)
        end   = datetime(args.year, 12, 31)
    else:
        start, end = date_range_from_data()
    log.info(f"Range: {start.date()} → {end.date()}")

    stations = build_pixel_indices(auth)
    log.info(f"{len(stations)} stations. Output: {PER_STA}")

    # Resume — for each station, find max already-cached timestamp; we walk
    # the range from the earliest "needs more data" station forward.
    earliest_resume = None
    for sta in stations["station"]:
        last = already_done_through(sta)
        if last is None:
            earliest_resume = start
            break
        if earliest_resume is None or last < earliest_resume:
            earliest_resume = last + timedelta(minutes=TIME_STEP_MIN)
    if earliest_resume and earliest_resume > start:
        log.info(f"  resuming from {earliest_resume:%Y-%m-%d %H:%M}")
        start = earliest_resume

    extract_range(start, end, stations, auth)
    log.info("Done.")


if __name__ == "__main__":
    main()
