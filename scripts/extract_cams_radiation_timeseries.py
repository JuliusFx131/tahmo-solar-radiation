"""
CAMS Solar Radiation Time-Series (cams-solar-radiation-timeseries) per station.

This is the *killer feature* the LB leader hinted at in the forum
(thisiskuhan, 2026-05-15): a per-station, 15-min-cadence pre-computed
estimate of GHI/BHI/DHI/BNI (all-sky AND clear-sky) for the exact lat/lon
of each station, covering 2004-present.

Source: Copernicus Atmosphere Data Store (ADS) — same auth as our existing
        CAMS aerosols (`ADS_KEY` + `ADS_URL` in _env.sh). Dataset has its
        own license: accept once on the dataset page if you haven't:
            https://ads.atmosphere.copernicus.eu/datasets/cams-solar-radiation-timeseries

Per-station API call returns a single CSV covering the date range we ask
for. We request 2016-01-01 → 2020-12-31 for each of the 40 stations and
save to per-station CSVs (resumable). Final concat → data/satellite/cams_radiation_ts.csv

Output columns (ext_csr_* prefix; "csr" = CAMS solar radiation):
  ext_csr_ghi              — All-sky Global Horizontal Irradiance (W/m²)
  ext_csr_bhi              — All-sky Beam Horizontal Irradiance
  ext_csr_dhi              — All-sky Diffuse Horizontal Irradiance
  ext_csr_bni              — All-sky Beam Normal Irradiance
  ext_csr_clearsky_ghi
  ext_csr_clearsky_bhi
  ext_csr_clearsky_dhi
  ext_csr_clearsky_bni
  ext_csr_reliability      — flag {0,1,2}: 0=ok, 1=partial, 2=missing
  ext_csr_clearness_kt     — all-sky GHI / clear-sky GHI (derived)
  ext_csr_diffuse_fraction — DHI / GHI (derived; NaN when GHI = 0)

Run:
  bash /workspace/shell/run_cams_radiation_ts.sh

Resumable: per-station CSVs cached under
  data/satellite/cams_radiation_per_station/
"""

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT / "data" / "raw"
SAT_DIR  = ROOT / "data" / "satellite"
PER_STA  = SAT_DIR / "cams_radiation_per_station"
PER_STA.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = RAW_DIR / "Train.csv"
TEST_CSV  = RAW_DIR / "Test.csv"
OUT       = SAT_DIR / "cams_radiation_ts.csv"


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
    return full["ts"].min().strftime("%Y-%m-%d"), full["ts"].max().strftime("%Y-%m-%d")


def fetch_station(client, lat: float, lon: float, alt: float,
                  d_start: str, d_end: str, out_path: Path):
    """Request 15-min radiation timeseries for one station."""
    # The ADS API for cams-solar-radiation-timeseries uses a slightly different
    # request shape than the EAC4 we used earlier — see ADS dataset docs.
    request = {
        "sky_type":       "observed_cloud",            # all-sky
        "location": {"latitude": float(lat), "longitude": float(lon)},
        "altitude":       str(float(alt)),
        "date":           f"{d_start}/{d_end}",
        "time_step":      "15minute",
        "time_reference": "universal_time",
        "format":         "csv",
    }
    client.retrieve("cams-solar-radiation-timeseries", request, str(out_path))


def parse_cams_csv(path: Path, station: str) -> pd.DataFrame:
    """CAMS solar-radiation CSV format (post-2019):
       - First lines are comments starting with '#'
       - The column header is ALSO commented:
           '# Observation period;TOA;Clear sky GHI;Clear sky BHI;Clear sky DHI;Clear sky BNI;GHI;BHI;DHI;BNI;Reliability'
       - Data rows: 'YYYY-MM-DDTHH:MM:SS.0/YYYY-MM-DDTHH:MM:SS.0;val;val;...'
       - Units are Wh/m² integrated over 15 minutes → multiply by 4 to get W/m² rate.
    """
    # Hardcode columns; comment lines vary but the data schema is fixed.
    cols = ["obs_period", "TOA",
            "Clear sky GHI", "Clear sky BHI", "Clear sky DHI", "Clear sky BNI",
            "GHI", "BHI", "DHI", "BNI", "Reliability"]
    df = pd.read_csv(path, sep=";", comment="#", header=None, names=cols)

    df["timestamp"] = pd.to_datetime(df["obs_period"].astype(str).str.split("/").str[0])

    rename = {
        "GHI":           "ext_csr_ghi",
        "BHI":           "ext_csr_bhi",
        "DHI":           "ext_csr_dhi",
        "BNI":           "ext_csr_bni",
        "Clear sky GHI": "ext_csr_clearsky_ghi",
        "Clear sky BHI": "ext_csr_clearsky_bhi",
        "Clear sky DHI": "ext_csr_clearsky_dhi",
        "Clear sky BNI": "ext_csr_clearsky_bni",
        "Reliability":   "ext_csr_reliability",
    }
    df = df.rename(columns=rename)

    # Convert Wh/m² per 15-min → W/m² (rate). Reliability stays as fraction.
    rate_cols = ["ext_csr_ghi", "ext_csr_bhi", "ext_csr_dhi", "ext_csr_bni",
                 "ext_csr_clearsky_ghi", "ext_csr_clearsky_bhi",
                 "ext_csr_clearsky_dhi", "ext_csr_clearsky_bni"]
    for c in rate_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32) * 4.0
    df["ext_csr_reliability"] = pd.to_numeric(df["ext_csr_reliability"], errors="coerce").astype(np.float32)

    # Derived
    cs = df["ext_csr_clearsky_ghi"].values
    g  = df["ext_csr_ghi"].values
    dh = df["ext_csr_dhi"].values
    df["ext_csr_clearness_kt"]     = np.where(cs > 5, g / cs, np.nan).astype(np.float32)
    df["ext_csr_diffuse_fraction"] = np.where(g  > 5, dh / g, np.nan).astype(np.float32)

    df["station"] = station
    keep = ["station", "timestamp"] + [v for v in rename.values()] + \
           ["ext_csr_clearness_kt", "ext_csr_diffuse_fraction"]
    return df[keep]


def main():
    import cdsapi

    t0 = time.time()
    ads_key = os.environ.get("ADS_KEY", "")
    ads_url = os.environ.get("ADS_URL", "https://ads.atmosphere.copernicus.eu/api")
    if not ads_key:
        log.error("ADS_KEY missing in _env.sh")
        return

    client = cdsapi.Client(url=ads_url, key=ads_key)
    stations = load_stations()
    log.info(f"Stations: {len(stations)}")

    files = []
    for i, row in stations.iterrows():
        sta = row["station"]
        per_path = PER_STA / f"{sta}.csv"
        if per_path.exists() and per_path.stat().st_size > 1000:
            log.info(f"  [{i+1:>2}/{len(stations)}] {sta} cached "
                     f"({per_path.stat().st_size:,} B)")
            files.append((sta, per_path))
            continue

        d_start, d_end = date_range_for_station(sta)
        log.info(f"  [{i+1:>2}/{len(stations)}] {sta} "
                 f"({row['latitude']:.3f}, {row['longitude']:.3f}, alt={row['elevation']:.0f}) "
                 f"{d_start} → {d_end}")
        try:
            fetch_station(client, row["latitude"], row["longitude"], row["elevation"],
                          d_start, d_end, per_path)
            log.info(f"      saved {per_path.stat().st_size:,} B → {per_path.name}")
            files.append((sta, per_path))
        except Exception as e:
            log.error(f"      FAILED: {e}")
            continue

    if not files:
        log.error("No per-station files. Nothing to merge.")
        return

    log.info(f"Parsing + concatenating {len(files)} per-station files ...")
    big_chunks = []
    for sta, path in files:
        try:
            big_chunks.append(parse_cams_csv(path, sta))
        except Exception as e:
            log.warning(f"  parse failed for {sta}: {e}")
    big = pd.concat(big_chunks, ignore_index=True)
    big = big.drop_duplicates(subset=["station", "timestamp"]).sort_values(["station", "timestamp"])
    # Put station first
    cols = ["station", "timestamp"] + [c for c in big.columns
                                        if c not in ("station", "timestamp")]
    big = big[cols]
    big.to_csv(OUT, index=False)
    log.info(f"Saved: {OUT}  ({len(big):,} rows × {len(big.columns)} cols)")
    log.info(f"Wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
