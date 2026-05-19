"""
Open-Meteo ERA5 archive — per-station hourly fetch.

Free API (https://archive.open-meteo.com/v1/archive), no auth required.
The leader on the LB (`fgbfgb`) reported these variables as useful, and we
were missing several relative to ECMWF CDS:
  • direct + diffuse radiation (we only had total ssrd)
  • cloud cover broken into low / mid / high layers (we only had total tcc)
  • wind (speed + direction)
  • CAPE (convective potential — proxy for thunderstorm activity)
  • dewpoint (better humidity / fog proxy than RH alone)

Output columns (ext_om_* prefix):
  ext_om_ghi               — shortwave_radiation (W/m²)
  ext_om_dni               — direct_normal_irradiance (W/m²)
  ext_om_dhi               — diffuse_radiation (W/m²)
  ext_om_direct_horiz      — direct_radiation on horizontal (W/m²)
  ext_om_cc_total          — cloud_cover total (%)
  ext_om_cc_low            — cloud_cover_low (%)
  ext_om_cc_mid            — cloud_cover_mid (%)
  ext_om_cc_high           — cloud_cover_high (%)
  ext_om_wind_speed_10m    — m/s
  ext_om_wind_dir_10m      — deg
  ext_om_cape              — J/kg
  ext_om_temperature_2m    — °C
  ext_om_dewpoint_2m       — °C
  ext_om_humidity_2m       — %
  ext_om_pressure_surface  — hPa
  ext_om_precip            — mm

Run:
  bash /workspace/shell/run_open_meteo.sh

Output:
  data/satellite/open_meteo_hourly.csv

Resumable: per-station checkpointing via individual per-station CSVs that are
concatenated at the end. If interrupted, rerun — already-fetched stations skip.
"""

import io
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
PER_STA  = SAT_DIR / "open_meteo_per_station"
PER_STA.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = RAW_DIR / "Train.csv"
TEST_CSV  = RAW_DIR / "Test.csv"
OUT       = SAT_DIR / "open_meteo_hourly.csv"

API = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
    "shortwave_radiation",
    "direct_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_direction_10m",
    "cape",
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "precipitation",
]

RENAME = {
    "shortwave_radiation":        "ext_om_ghi",
    "direct_radiation":           "ext_om_direct_horiz",
    "direct_normal_irradiance":   "ext_om_dni",
    "diffuse_radiation":          "ext_om_dhi",
    "cloud_cover":                "ext_om_cc_total",
    "cloud_cover_low":            "ext_om_cc_low",
    "cloud_cover_mid":            "ext_om_cc_mid",
    "cloud_cover_high":           "ext_om_cc_high",
    "wind_speed_10m":             "ext_om_wind_speed_10m",
    "wind_direction_10m":         "ext_om_wind_dir_10m",
    "cape":                       "ext_om_cape",
    "temperature_2m":             "ext_om_temperature_2m",
    "dew_point_2m":               "ext_om_dewpoint_2m",
    "relative_humidity_2m":       "ext_om_humidity_2m",
    "surface_pressure":           "ext_om_pressure_surface",
    "precipitation":              "ext_om_precip",
}

MAX_RETRIES = 6
RETRY_BACKOFF = 5
REQUEST_DELAY = 8.0    # 8 sec between stations stays well under free-tier burst limit
RATE_LIMIT_SLEEP = 90  # seconds to sleep specifically on HTTP 429


def load_stations() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_CSV)
    return (train.groupby("station")[["latitude", "longitude", "elevation"]]
            .first().reset_index())


def date_range_for_station(station: str) -> tuple[str, str]:
    """Pull the min/max timestamp covered by train+test for this station."""
    df_tr = pd.read_csv(TRAIN_CSV, usecols=["station", "timestamp"])
    df_te = pd.read_csv(TEST_CSV,  usecols=["station", "timestamp"])
    full = pd.concat([df_tr, df_te])
    full = full[full["station"] == station]
    full["ts"] = pd.to_datetime(full["timestamp"], format="mixed", dayfirst=True)
    return full["ts"].min().strftime("%Y-%m-%d"), full["ts"].max().strftime("%Y-%m-%d")


def _retry(fn, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except requests.HTTPError as e:
            if attempt == MAX_RETRIES - 1:
                raise
            status = getattr(e.response, "status_code", None)
            if status == 429:
                # Rate-limited — needs a long cool-down, not exponential backoff
                wait = RATE_LIMIT_SLEEP * (attempt + 1)
                log.warning(f"  HTTP 429 — rate-limited, sleeping {wait}s "
                            f"(retry {attempt + 1}/{MAX_RETRIES})")
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


def fetch_station(station: str, lat: float, lon: float, elev: float,
                  date_start: str, date_end: str) -> pd.DataFrame:
    params = {
        "latitude":     lat,
        "longitude":    lon,
        "elevation":    elev,
        "start_date":   date_start,
        "end_date":     date_end,
        "hourly":       ",".join(HOURLY_VARS),
        "timezone":     "GMT",
    }
    resp = _retry(requests.get, API, params=params, timeout=90)
    resp.raise_for_status()
    js = resp.json()
    if "hourly" not in js:
        raise RuntimeError(f"no 'hourly' in response: {list(js.keys())} — {js.get('reason','')}")
    h = js["hourly"]
    df = pd.DataFrame({k: h[k] for k in ["time"] + HOURLY_VARS})
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"]).rename(columns=RENAME)
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
                               float(row["elevation"]), d_start, d_end)
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
