"""
NASA POWER (MERRA-2 + GEOS satellite correction) — hourly per-station fetch.

Free HTTPS API, no auth required.
https://power.larc.nasa.gov/docs/services/api/temporal/hourly/

Gives us a 4th independent radiation estimate (after Solar/Open-Meteo/ERA5):
NASA POWER blends MERRA-2 reanalysis with GEOS satellite cloud correction, so
it's a DIFFERENT model family than the ECMWF (Open-Meteo/ERA5) line. That
independence is the main reason to add it — strong ensemble diversity.

Variables (renamed with ext_np_ prefix):
  ALLSKY_SFC_SW_DWN  → ext_np_allsky_ghi      (W/m²)
  ALLSKY_SFC_SW_DIFF → ext_np_allsky_dhi      (W/m²)
  ALLSKY_SFC_SW_DNI  → ext_np_allsky_dni      (W/m²)
  CLRSKY_SFC_SW_DWN  → ext_np_clrsky_ghi      (W/m²)
  ALLSKY_KT          → ext_np_clearness_index (0-1)
  CLOUD_AMT          → ext_np_cloud_amount    (%)
  AOD_55             → ext_np_aod_550         (unitless)
  PRECTOTCORR        → ext_np_precip_corr     (mm/hr, bias-corrected)

Run:
  bash /workspace/shell/run_nasa_power.sh

Resumable: per-station CSVs cached under
  data/satellite/nasa_power_per_station/

Final concatenated output:
  data/satellite/nasa_power_hourly.csv
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT / "data" / "raw"
SAT_DIR  = ROOT / "data" / "satellite"
PER_STA  = SAT_DIR / "nasa_power_per_station"
PER_STA.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = RAW_DIR / "Train.csv"
TEST_CSV  = RAW_DIR / "Test.csv"
OUT       = SAT_DIR / "nasa_power_hourly.csv"

API = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# Hourly community=RE (Renewable Energy) parameter set
PARAMS = [
    "ALLSKY_SFC_SW_DWN",
    "ALLSKY_SFC_SW_DIFF",
    "ALLSKY_SFC_SW_DNI",
    "CLRSKY_SFC_SW_DWN",
    "ALLSKY_KT",
    "CLOUD_AMT",
    "AOD_55",
    "PRECTOTCORR",
]

RENAME = {
    "ALLSKY_SFC_SW_DWN":  "ext_np_allsky_ghi",
    "ALLSKY_SFC_SW_DIFF": "ext_np_allsky_dhi",
    "ALLSKY_SFC_SW_DNI":  "ext_np_allsky_dni",
    "CLRSKY_SFC_SW_DWN":  "ext_np_clrsky_ghi",
    "ALLSKY_KT":          "ext_np_clearness_index",
    "CLOUD_AMT":          "ext_np_cloud_amount",
    "AOD_55":             "ext_np_aod_550",
    "PRECTOTCORR":        "ext_np_precip_corr",
}

MAX_RETRIES   = 6
RETRY_BACKOFF = 5
RATE_LIMIT_SLEEP = 60
REQUEST_DELAY = 4.0    # NASA POWER says 1 req/sec is safe; we go slower to be polite


def load_stations() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_CSV)
    return (train.groupby("station")[["latitude", "longitude", "elevation"]]
            .first().reset_index())


def date_range_for_station(station: str) -> tuple[str, str]:
    df_tr = pd.read_csv(TRAIN_CSV, usecols=["station", "timestamp"])
    df_te = pd.read_csv(TEST_CSV,  usecols=["station", "timestamp"])
    full = pd.concat([df_tr, df_te])
    full = full[full["station"] == station]
    full["ts"] = pd.to_datetime(full["timestamp"], format="mixed", dayfirst=True)
    return full["ts"].min().strftime("%Y%m%d"), full["ts"].max().strftime("%Y%m%d")


def _retry(fn, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except requests.HTTPError as e:
            if attempt == MAX_RETRIES - 1:
                raise
            status = getattr(e.response, "status_code", None)
            if status == 429:
                wait = RATE_LIMIT_SLEEP * (attempt + 1)
                log.warning(f"  HTTP 429 — rate-limited, sleeping {wait}s")
            else:
                wait = RETRY_BACKOFF * (2 ** attempt)
                log.warning(f"  retry {attempt + 1}/{MAX_RETRIES} after {wait}s: HTTP {status}")
            time.sleep(wait)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = RETRY_BACKOFF * (2 ** attempt)
            log.warning(f"  retry {attempt + 1}/{MAX_RETRIES} after {wait}s: {e}")
            time.sleep(wait)


def fetch_station(station: str, lat: float, lon: float, d_start: str, d_end: str) -> pd.DataFrame:
    params = {
        "parameters": ",".join(PARAMS),
        "community":  "RE",
        "longitude":  lon,
        "latitude":   lat,
        "start":      d_start,
        "end":        d_end,
        "format":     "JSON",
    }
    resp = _retry(requests.get, API, params=params, timeout=120)
    resp.raise_for_status()
    js = resp.json()
    # The hourly response packs each parameter as {datetime_str: value} dict
    block = js["properties"]["parameter"]
    # Index by timestamp from the first parameter we requested
    any_key = next(iter(block))
    times = sorted(block[any_key].keys())
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(times, format="%Y%m%d%H"),
    })
    for p in PARAMS:
        col = RENAME[p]
        df[col] = [block[p].get(t, np.nan) for t in times]
    # NASA POWER fill values
    df = df.replace(-999.0, np.nan)
    df.insert(0, "station", station)
    return df


def main():
    t0 = time.time()
    stations = load_stations()
    log.info(f"Stations: {len(stations)}")

    fetched = []
    for i, row in stations.iterrows():
        sta = row["station"]
        per_path = PER_STA / f"{sta}.csv"
        if per_path.exists() and per_path.stat().st_size > 1000:
            log.info(f"  [{i+1:>2}/{len(stations)}] {sta} cached ({per_path.stat().st_size:,} B)")
            fetched.append(per_path)
            continue

        d_start, d_end = date_range_for_station(sta)
        log.info(f"  [{i+1:>2}/{len(stations)}] {sta} "
                 f"({row['latitude']:.3f}, {row['longitude']:.3f}) "
                 f"{d_start} → {d_end}")
        try:
            df = fetch_station(sta, float(row["latitude"]), float(row["longitude"]),
                               d_start, d_end)
            df.to_csv(per_path, index=False)
            log.info(f"      saved {len(df):,} hourly rows → {per_path.name}")
            fetched.append(per_path)
        except Exception as e:
            log.error(f"      FAILED: {e}")
            continue
        time.sleep(REQUEST_DELAY)

    if not fetched:
        log.error("No per-station files. Nothing to merge.")
        return

    log.info(f"Concatenating {len(fetched)} per-station files ...")
    big = pd.concat([pd.read_csv(p, parse_dates=["timestamp"]) for p in fetched],
                    ignore_index=True)
    big = big.drop_duplicates(subset=["station", "timestamp"]).sort_values(["station", "timestamp"])
    big.to_csv(OUT, index=False)
    log.info(f"Saved: {OUT}  ({len(big):,} rows × {len(big.columns)} cols)")
    log.info(f"Wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
