"""
TAHMO Satellite Data Extraction Pipeline
=========================================
Downloads, computes, and merges external data sources into Train/Test CSVs.

All external columns are prefixed for easy filtering:
  ext_sol_*   — computed solar geometry (no download needed)
  ext_lsa_*   — LSA-SAF DSSF (EUMETSAT)
  ext_tro_*   — TROPOMI cloud + aerosol (Copernicus Data Space)
  ext_era5_*  — ERA5 reanalysis (CDS)
  ext_cams_*  — CAMS aerosol optical depth (CDS)
  ext_modis_* — MODIS cloud + albedo (NASA Earthdata)
  ext_msg_*   — MSG cloud physical properties (EUMETSAT)

Drop all external:    df[[c for c in df.columns if not c.startswith('ext_')]]
Drop one source:      df.drop(columns=[c for c in df.columns if c.startswith('ext_era5_')])

Install dependencies:
  pip install eumdac netCDF4 h5py xarray scipy requests tqdm cdsapi

Usage:
  python data_pipeline.py --source all  ... (all credentials)
  python data_pipeline.py --source lsa_saf --eumetsat-key KEY --eumetsat-secret SECRET
  python data_pipeline.py --source tropomi --cdse-user USER --cdse-password PASS
  python data_pipeline.py --source era5 --cds-key CDS_API_KEY
  python data_pipeline.py --source cams --cds-key CDS_API_KEY
  python data_pipeline.py --source modis --earthdata-token TOKEN
  python data_pipeline.py --source msg_cloud --eumetsat-key KEY --eumetsat-secret SECRET
  python data_pipeline.py --source solar   (no credentials needed)
  python data_pipeline.py --merge-only     (just merges already-downloaded CSVs)

Account registration:
  EUMETSAT:    https://eoportal.eumetsat.int/
  Copernicus:  https://dataspace.copernicus.eu/
  CDS:         https://cds.climate.copernicus.eu/
  NASA:        https://urs.earthdata.nasa.gov/

Final outputs:
  data/processed/Train_enhanced.csv
  data/processed/Test_enhanced.csv
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
RAW_DIR     = ROOT / "data" / "raw"
PROC_DIR    = ROOT / "data" / "processed"
SAT_DIR     = ROOT / "data" / "satellite"
SAT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV   = RAW_DIR / "Train.csv"
TEST_CSV    = RAW_DIR / "Test.csv"

ALL_SOURCES = ["solar", "lsa_saf", "tropomi", "era5", "cams"]
# msg_cloud excluded: EO:EUM:DAT:MSG:CLP removed from EUMETSAT Data Store (404)

# ─── Retry helper ────────────────────────────────────────────────────────────

MAX_RETRIES = 4
RETRY_BACKOFF = 2


def _retry(fn, *args, retries=MAX_RETRIES, **kwargs):
    """Call fn with exponential backoff on failure."""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = RETRY_BACKOFF * (2 ** attempt)
            log.warning(f"  Retry {attempt + 1}/{retries} after {wait}s: {e}")
            time.sleep(wait)


# ─── Atomic checkpoint I/O ───────────────────────────────────────────────────
# Disk-full mid-write to_csv() leaves a 0-byte file and crashes the next read.
# Write to a .tmp sibling then atomic rename — failure preserves the prior good
# checkpoint. Read side tolerates empty/corrupt files (treats them as absent).

def _safe_write_csv(records, path):
    """Atomically write a DataFrame-able records list to path."""
    df = pd.DataFrame(records)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_csv(tmp, index=False)
        tmp.replace(path)
    except OSError as e:
        log.warning(f"  checkpoint write failed ({e}); previous {path.name} is intact")
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _safe_read_csv(path, **kwargs):
    """Read a checkpoint CSV; return None if missing/empty/corrupt."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        log.warning(f"  {path.name} corrupt ({e}); discarding and starting fresh")
        return None


# ─── Station metadata ─────────────────────────────────────────────────────────

def load_stations() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_CSV)
    stations = (
        train.groupby("station")[["latitude", "longitude", "elevation", "country"]]
        .first()
        .reset_index()
    )
    log.info(f"Loaded {len(stations)} stations across "
             f"lat [{stations.latitude.min():.1f}, {stations.latitude.max():.1f}] "
             f"lon [{stations.longitude.min():.1f}, {stations.longitude.max():.1f}]")
    return stations


def load_timestamps() -> pd.DataFrame:
    """Return all unique (station, timestamp) pairs from train + test."""
    train = pd.read_csv(TRAIN_CSV, usecols=["station", "timestamp"])
    test  = pd.read_csv(TEST_CSV,  usecols=["station", "timestamp"])
    combined = pd.concat([train, test], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], format="mixed", dayfirst=True)
    return combined


def _station_bbox(stations: pd.DataFrame, buffer: float = 0.5) -> dict:
    """Bounding box around all stations with buffer in degrees."""
    return {
        "lat_min": stations["latitude"].min()  - buffer,
        "lat_max": stations["latitude"].max()  + buffer,
        "lon_min": stations["longitude"].min() - buffer,
        "lon_max": stations["longitude"].max() + buffer,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 1: SOLAR GEOMETRY (computed — no download needed)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_solar_features() -> None:
    """
    Compute solar position and clear-sky radiation for every (station, timestamp)
    in train + test. Pure math — no API or download needed.

    Output columns (ext_sol_* prefix):
      ext_sol_zenith        — solar zenith angle (degrees, 90 = horizon)
      ext_sol_azimuth       — solar azimuth angle (degrees, 0 = north)
      ext_sol_elevation     — solar elevation angle (degrees, 0 = horizon)
      ext_sol_hour_angle    — hour angle (degrees)
      ext_sol_declination   — solar declination (degrees)
      ext_sol_eqtime        — equation of time (minutes)
      ext_sol_earth_sun_dist — earth-sun distance factor (AU ratio squared)
      ext_sol_clearsky       — estimated clear-sky radiation (W/m²)
      ext_sol_daylight       — 1 if sun is above horizon, 0 otherwise
      ext_sol_day_length     — hours of daylight for this date+latitude
    """
    log.info("=== Solar Geometry Features (computed) ===")

    ts_df = load_timestamps()
    stations = load_stations()

    merged = ts_df.merge(
        stations[["station", "latitude", "longitude", "elevation"]],
        on="station", how="left"
    )

    ts = merged["timestamp"]
    lat = np.radians(merged["latitude"].values)
    lon = merged["longitude"].values
    elev = merged["elevation"].values

    # Day of year and fractional year
    doy = ts.dt.dayofyear.values.astype(float)
    hour = ts.dt.hour.values + ts.dt.minute.values / 60.0
    gamma = 2 * np.pi * (doy - 1) / 365.0  # fractional year in radians

    # Equation of time (minutes)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)   - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2*gamma) - 0.04089  * np.sin(2*gamma)
    )

    # Solar declination (radians)
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)   + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2*gamma) + 0.000907 * np.sin(2*gamma)
        - 0.002697 * np.cos(3*gamma) + 0.00148  * np.sin(3*gamma)
    )

    # Solar time
    time_offset = eqtime + 4 * lon  # minutes
    true_solar_time = hour * 60 + time_offset
    hour_angle = np.radians((true_solar_time / 4) - 180)

    # Solar zenith angle
    cos_zenith = (np.sin(lat) * np.sin(decl) +
                  np.cos(lat) * np.cos(decl) * np.cos(hour_angle))
    cos_zenith = np.clip(cos_zenith, -1, 1)
    zenith = np.degrees(np.arccos(cos_zenith))

    # Solar azimuth
    sin_azimuth = -np.cos(decl) * np.sin(hour_angle) / np.sin(np.radians(zenith) + 1e-10)
    sin_azimuth = np.clip(sin_azimuth, -1, 1)
    azimuth = np.degrees(np.arcsin(sin_azimuth))

    # Solar elevation
    elevation_angle = 90 - zenith

    # Earth-sun distance factor
    earth_sun_dist = (
        1.000110
        + 0.034221 * np.cos(gamma)   + 0.001280 * np.sin(gamma)
        + 0.000719 * np.cos(2*gamma) + 0.000077 * np.sin(2*gamma)
    )

    # Clear-sky radiation estimate (simplified Ineichen-Perez model)
    SOLAR_CONSTANT = 1361.0  # W/m²
    # Altitude correction for atmospheric thickness
    altitude_km = np.clip(elev, 0, 5000) / 1000.0
    am = 1 / (cos_zenith + 0.50572 * np.power(96.07995 - zenith + 1e-10, -1.6364) + 1e-10)
    am = np.clip(am, 1, 40)
    # Simplified clear-sky: Beer-Lambert with altitude correction
    clearsky = SOLAR_CONSTANT * earth_sun_dist * cos_zenith * np.exp(
        -0.09 * am * np.exp(-0.00013 * elev)
    )
    clearsky = np.where(cos_zenith <= 0, 0, clearsky)
    clearsky = np.clip(clearsky, 0, 1400)

    # Day length (hours)
    cos_ha_sunset = -np.tan(lat) * np.tan(decl)
    cos_ha_sunset = np.clip(cos_ha_sunset, -1, 1)
    day_length = 2 * np.degrees(np.arccos(cos_ha_sunset)) / 15.0

    # Build output
    result = pd.DataFrame({
        "station":                merged["station"],
        "timestamp":              merged["timestamp"],
        "ext_sol_zenith":         np.round(zenith, 2),
        "ext_sol_azimuth":        np.round(azimuth, 2),
        "ext_sol_elevation":      np.round(elevation_angle, 2),
        "ext_sol_hour_angle":     np.round(np.degrees(hour_angle), 2),
        "ext_sol_declination":    np.round(np.degrees(decl), 4),
        "ext_sol_eqtime":         np.round(eqtime, 2),
        "ext_sol_earth_sun_dist": np.round(earth_sun_dist, 6),
        "ext_sol_clearsky":       np.round(clearsky, 1),
        "ext_sol_daylight":       (elevation_angle > 0).astype(int),
        "ext_sol_day_length":     np.round(day_length, 2),
    })

    out = SAT_DIR / "solar_features.csv"
    result.to_csv(out, index=False)
    log.info(f"Saved solar features: {out}  ({len(result):,} rows, "
             f"{len([c for c in result.columns if c.startswith('ext_')])} columns)")


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 2: SARAH-3 Surface Radiation via EUMETSAT Data Store
# ═══════════════════════════════════════════════════════════════════════════════

def extract_lsa_saf(consumer_key: str, consumer_secret: str) -> pd.DataFrame:
    """
    Download SARAH-3 (Surface Radiation Data Set - Heliosat) daily surface
    solar radiation for all station locations.

    Collection: EO:EUM:DAT:0863
    Products used:
      SISdm — Surface Incoming Shortwave daily mean (W/m²)
      SIDdm — Surface direct Irradiance daily mean (W/m²)
      DNIdm — Direct Normal Irradiance daily mean (W/m²)

    Note: replaces the old EO:EUM:DAT:MSG:DSSF which is no longer available.

    Output columns: ext_lsa_sis, ext_lsa_sid, ext_lsa_dni
    """
    try:
        import eumdac
    except ImportError:
        log.error("pip install eumdac")
        sys.exit(1)
    try:
        import xarray as xr
    except ImportError:
        log.error("pip install xarray")
        sys.exit(1)

    log.info("=== SARAH-3 Surface Radiation Extraction ===")

    token = eumdac.AccessToken((consumer_key, consumer_secret))
    log.info(f"EUMETSAT token expires: {token.expiration}")
    datastore = eumdac.DataStore(token)
    stations = load_stations()

    ts_df = load_timestamps()
    date_min = ts_df["timestamp"].dt.date.min()
    date_max = ts_df["timestamp"].dt.date.max()

    collection = datastore.get_collection("EO:EUM:DAT:0863")
    log.info(f"Collection: {collection.title}")

    checkpoint_path = SAT_DIR / "sarah_checkpoint.csv"
    existing = _safe_read_csv(checkpoint_path)
    if existing is not None:
        records = existing.to_dict("records")
        log.info(f"Resumed checkpoint: {len(records):,} records")
    else:
        records = []

    # Process day by day, download SISdm, SIDdm, DNIdm products
    current = datetime(date_min.year, date_min.month, 1)
    end = datetime(date_max.year, date_max.month, date_max.day)
    delta = timedelta(days=1)

    while current <= end:
        try:
            products = list(collection.search(
                dtstart=current, dtend=current + delta
            ))

            # Filter for daily mean products we want
            target_prefixes = {"SISdm", "SIDdm", "DNIdm"}
            day_data = {}

            for product in products:
                pid = str(product)
                prefix = pid[:5]
                if prefix not in target_prefixes:
                    continue
                if prefix in day_data:
                    continue

                try:
                    import io, zipfile, tempfile
                    with product.open() as fsrc:
                        data_bytes = fsrc.read()

                    # SARAH-3 payload is a ZIP containing one .nc — extract it.
                    if data_bytes[:4] == b"PK\x03\x04":
                        with zipfile.ZipFile(io.BytesIO(data_bytes)) as zf:
                            nc_name = next((n for n in zf.namelist() if n.endswith(".nc")), None)
                            if nc_name is None:
                                raise RuntimeError(f"No .nc inside {pid}")
                            nc_bytes = zf.read(nc_name)
                    else:
                        nc_bytes = data_bytes

                    # netCDF4 backend needs a real file path; xarray's lazy
                    # loading means the file must outlive the ds. Use a
                    # persistent temp file and clean up after extraction.
                    tf = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
                    tf.write(nc_bytes); tf.close()
                    try:
                        ds = xr.open_dataset(tf.name, engine="netcdf4")
                        for _, sta in stations.iterrows():
                            point = ds.sel(
                                lat=sta["latitude"],
                                lon=sta["longitude"],
                                method="nearest",
                                tolerance=0.5,
                            )
                            var_names = [v for v in ds.data_vars if v in ("SIS", "SID", "DNI")]
                            for var in var_names:
                                val = float(point[var].values)
                                key = (sta["station"], current.date())
                                if key not in day_data:
                                    day_data[key] = {
                                        "station": sta["station"],
                                        "date": current.date(),
                                    }
                                col_name = f"ext_lsa_{var.lower()}"
                                if val < 0 or val > 1400:
                                    val = np.nan
                                day_data[key][col_name] = val
                        ds.close()
                    finally:
                        import os as _os
                        try: _os.unlink(tf.name)
                        except FileNotFoundError: pass
                except Exception as e:
                    log.warning(f"  SARAH {pid}: {e}")

            records.extend(day_data.values())

        except Exception as e:
            log.debug(f"  SARAH {current:%Y-%m-%d}: {e}")

        current += delta
        if current.day == 1:
            log.info(f"  SARAH completed {(current - delta):%Y-%m}")
            if records:
                _safe_write_csv(records, checkpoint_path)

    df = pd.DataFrame(records)
    if df.empty:
        log.error("No SARAH data extracted.")
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["station", "date"])
    df = df.sort_values(["station", "date"]).reset_index(drop=True)

    out = SAT_DIR / "sarah_radiation.csv"
    df.to_csv(out, index=False)
    log.info(f"Saved: {out}  ({len(df):,} rows)")

    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 3: TROPOMI (Sentinel-5P) via Copernicus Data Space
# ═══════════════════════════════════════════════════════════════════════════════

def extract_tropomi(cdse_user: str, cdse_password: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download TROPOMI L2 cloud fraction + UV aerosol index, 2018-05 to 2020-12.
    Output columns: ext_tro_cloud_fraction, ext_tro_cloud_top_pressure,
                    ext_tro_aerosol_index, ext_tro_aerosol_index_354_388
    """
    try:
        import requests
    except ImportError:
        log.error("pip install requests")
        sys.exit(1)

    log.info("=== TROPOMI Extraction ===")
    stations = load_stations()
    bbox = _station_bbox(stations)
    log.info(f"Bounding box: lat [{bbox['lat_min']:.2f}, {bbox['lat_max']:.2f}], "
             f"lon [{bbox['lon_min']:.2f}, {bbox['lon_max']:.2f}]")

    token_state = {
        "token": _get_cdse_token(cdse_user, cdse_password),
        "obtained_at": time.time(),
    }

    def get_valid_token() -> str:
        if time.time() - token_state["obtained_at"] > 480:
            log.info("  Refreshing CDSE token...")
            token_state["token"] = _get_cdse_token(cdse_user, cdse_password)
            token_state["obtained_at"] = time.time()
        return token_state["token"]

    # CDSE OData productType values (no mission/processing prefix).
    # Verified against catalogue.dataspace.copernicus.eu 2026-05.
    products_to_fetch = [
        ("L2__CLOUD_",  "cloud",   ["cloud_fraction_crb",
                                     "cloud_top_pressure_crb"]),
        ("L2__AER_AI",  "aerosol", ["absorbing_aerosol_index",
                                     "aerosol_index_354_388"]),
    ]

    cloud_checkpoint   = SAT_DIR / "tropomi_cloud_checkpoint.csv"
    aerosol_checkpoint = SAT_DIR / "tropomi_aerosol_checkpoint.csv"

    cloud_records = _safe_read_csv(cloud_checkpoint)
    cloud_records = cloud_records.to_dict("records") if cloud_records is not None else []
    aerosol_records = _safe_read_csv(aerosol_checkpoint)
    aerosol_records = aerosol_records.to_dict("records") if aerosol_records is not None else []

    start = datetime(2018, 5, 1)
    if cloud_records or aerosol_records:
        all_dates = [pd.Timestamp(r["date"]) for r in cloud_records + aerosol_records]
        if all_dates:
            resume_date = max(all_dates).to_pydatetime() + timedelta(days=1)
            if resume_date > start:
                start = resume_date
                log.info(f"Resuming from {start:%Y-%m-%d}")

    end   = datetime(2020, 12, 31)
    delta = timedelta(days=1)

    # Subsample granules per day to keep total runtime reasonable.
    # Africa is covered by 1-3 distinct overpasses per day; 3 captures the
    # main passes while cutting ~5× off full-redundancy downloads.
    import os as _os
    MAX_GRANULES_PER_DAY = int(_os.environ.get("TROPOMI_MAX_GRANULES_PER_DAY", "3"))

    current = start
    while current <= end:
        day_had_records = False
        for product_type, label, variables in products_to_fetch:
            try:
                token = get_valid_token()
                granules = _retry(_search_cdse, token, product_type, current, bbox)
                if not granules:
                    continue
                granules = granules[:MAX_GRANULES_PER_DAY]

                for granule_id, granule_name, download_url in granules:
                    token = get_valid_token()
                    data_bytes = _retry(_download_cdse_product, token, download_url)
                    if data_bytes is None:
                        continue

                    import io, netCDF4 as nc4
                    with nc4.Dataset("in-memory", memory=data_bytes) as ds:
                        rows = _extract_tropomi_at_stations(ds, stations, variables, label, current)
                        if label == "cloud":
                            cloud_records.extend(rows)
                        else:
                            aerosol_records.extend(rows)
                        if rows:
                            day_had_records = True
            except Exception as e:
                log.warning(f"  {current:%Y-%m-%d} {label}: {e}")

        # Per-day progress log + per-day checkpoint write (atomic).
        # Keeps loss to <1 day if the process dies.
        log.info(f"  TROPOMI {current:%Y-%m-%d}  cloud_rows={len(cloud_records):,}  "
                 f"aerosol_rows={len(aerosol_records):,}")
        if cloud_records:
            _safe_write_csv(cloud_records, cloud_checkpoint)
        if aerosol_records:
            _safe_write_csv(aerosol_records, aerosol_checkpoint)

        current += delta
        if current.day == 1:
            log.info(f"  ===== Completed {(current - delta):%Y-%m} =====")
            if cloud_records:
                _safe_write_csv(cloud_records, cloud_checkpoint)
            if aerosol_records:
                _safe_write_csv(aerosol_records, aerosol_checkpoint)

    cloud_df   = _build_daily_df(cloud_records,   "cloud",   stations)
    aerosol_df = _build_daily_df(aerosol_records, "aerosol", stations)

    # Rename columns with ext_tro_ prefix
    cloud_rename = {
        "cloud_fraction_crb":     "ext_tro_cloud_fraction",
        "cloud_top_pressure_crb": "ext_tro_cloud_top_pressure",
    }
    aerosol_rename = {
        "absorbing_aerosol_index":  "ext_tro_aerosol_index",
        "aerosol_index_354_388":    "ext_tro_aerosol_index_354_388",
    }
    cloud_df   = cloud_df.rename(columns=cloud_rename)
    aerosol_df = aerosol_df.rename(columns=aerosol_rename)

    cloud_out   = SAT_DIR / "tropomi_cloud.csv"
    aerosol_out = SAT_DIR / "tropomi_aerosol.csv"
    cloud_df.to_csv(cloud_out, index=False)
    aerosol_df.to_csv(aerosol_out, index=False)
    log.info(f"Saved: {cloud_out}  ({len(cloud_df):,} rows)")
    log.info(f"Saved: {aerosol_out} ({len(aerosol_df):,} rows)")

    for cp in [cloud_checkpoint, aerosol_checkpoint]:
        if cp.exists():
            cp.unlink()

    return cloud_df, aerosol_df


def _get_cdse_token(user: str, password: str) -> str:
    import requests
    resp = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={"grant_type": "password", "client_id": "cdse-public",
              "username": user, "password": password},
        timeout=30,
    )
    resp.raise_for_status()
    log.info("CDSE: authenticated")
    return resp.json()["access_token"]


def _search_cdse(token: str, product_type: str, date: datetime, bbox: dict) -> list[tuple]:
    import requests
    date_from = date.strftime("%Y-%m-%dT00:00:00.000Z")
    date_to   = date.strftime("%Y-%m-%dT23:59:59.999Z")
    wkt = (f"POLYGON(({bbox['lon_min']} {bbox['lat_min']},"
           f"{bbox['lon_max']} {bbox['lat_min']},"
           f"{bbox['lon_max']} {bbox['lat_max']},"
           f"{bbox['lon_min']} {bbox['lat_max']},"
           f"{bbox['lon_min']} {bbox['lat_min']}))")
    url = (
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
        f"$filter=Collection/Name eq 'SENTINEL-5P' "
        f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        f"  and att/OData.CSC.StringAttribute/Value eq '{product_type}') "
        f"and ContentDate/Start gt {date_from} "
        f"and ContentDate/Start lt {date_to} "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')"
        f"&$top=20&$expand=Assets"
    )
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    resp.raise_for_status()
    return [(item["Id"], item["Name"],
             f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({item['Id']})/$value")
            for item in resp.json().get("value", [])]


def _download_cdse_product(token: str, url: str) -> bytes | None:
    import requests, zipfile, io
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=120, stream=True)
    resp.raise_for_status()
    data = resp.content
    if data[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            nc_files = [n for n in zf.namelist() if n.endswith(".nc")]
            if nc_files:
                return zf.read(nc_files[0])
    return data


def _extract_tropomi_at_stations(ds, stations, variables, label, date):
    from scipy.spatial import cKDTree
    records = []
    try:
        grp = ds["PRODUCT"]
        lat = grp["latitude"][0].data.ravel()
        lon = grp["longitude"][0].data.ravel()
        qa  = grp["qa_value"][0].data.ravel()
        good = (qa >= 0.5) & np.isfinite(lat) & np.isfinite(lon)
        if good.sum() == 0:
            return records
        tree = cKDTree(np.column_stack([lat[good], lon[good]]))
        var_data = {}
        for var in variables:
            try:
                arr = grp[var][0].data.ravel().astype(float)
                fill = grp[var]._FillValue if hasattr(grp[var], "_FillValue") else -999
                arr[arr == fill] = np.nan
                var_data[var] = arr[good]
            except Exception:
                var_data[var] = np.full(good.sum(), np.nan)
        for _, sta in stations.iterrows():
            dist, idx = tree.query([sta["latitude"], sta["longitude"]], k=1)
            if dist > 0.1:
                continue
            rec = {"station": sta["station"], "date": date.date()}
            for var in variables:
                rec[var] = float(var_data[var][idx])
            records.append(rec)
    except Exception as e:
        log.debug(f"TROPOMI extraction error: {e}")
    return records


def _build_daily_df(records, label, stations):
    if not records:
        log.warning(f"No {label} records")
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["station", "date"])
    df = df.sort_values(["station", "date"]).reset_index(drop=True)
    log.info(f"{label}: {len(df):,} rows, {df['station'].nunique()} stations")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 4: ERA5 Reanalysis via Copernicus CDS API
# ═══════════════════════════════════════════════════════════════════════════════

def extract_era5(cds_key: str) -> pd.DataFrame:
    """
    Download ERA5 hourly surface variables for all station locations.
    Uses the CDS API (pip install cdsapi).

    Output columns:
      ext_era5_ssrd         — surface solar radiation downwards (J/m², cumulative per hour)
      ext_era5_tcc          — total cloud cover (0-1)
      ext_era5_tcwv         — total column water vapour (kg/m²)
      ext_era5_blh          — boundary layer height (m)
      ext_era5_sp           — surface pressure (Pa)
    """
    try:
        import cdsapi
    except ImportError:
        log.error("pip install cdsapi")
        sys.exit(1)

    log.info("=== ERA5 Reanalysis Extraction ===")

    stations = load_stations()
    bbox = _station_bbox(stations, buffer=1.0)

    # CDS API client
    client = cdsapi.Client(key=cds_key)

    variables = [
        "surface_solar_radiation_downwards",  # J/m² cumulative → convert to W/m²
        "total_cloud_cover",
        "total_column_water_vapour",
        "boundary_layer_height",
        "surface_pressure",
    ]

    # Determine date range from train + test
    ts_df = load_timestamps()
    date_min = ts_df["timestamp"].dt.date.min()
    date_max = ts_df["timestamp"].dt.date.max()
    log.info(f"Date range: {date_min} to {date_max}")

    checkpoint_path = SAT_DIR / "era5_checkpoint.csv"
    existing = _safe_read_csv(checkpoint_path)
    if existing is not None:
        all_records = existing.to_dict("records")
        log.info(f"Resumed ERA5 checkpoint: {len(all_records):,} records")
    else:
        all_records = []

    # Download month by month
    current = datetime(date_min.year, date_min.month, 1)
    end = datetime(date_max.year, date_max.month, 1)

    while current <= end:
        month_str = current.strftime("%Y-%m")
        log.info(f"  Downloading ERA5 {month_str} ...")

        days_in_month = ((current.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)).day
        out_file = SAT_DIR / f"era5_{month_str}.nc"

        # Defensive: a truncated cached file (e.g. from prior disk-full) would
        # otherwise skip the re-download AND fail extraction every run.
        if out_file.exists() and out_file.stat().st_size < 5_000_000:
            log.warning(f"  cached {out_file.name} suspiciously small "
                        f"({out_file.stat().st_size:,}B), deleting for re-download")
            out_file.unlink()

        try:
            if not out_file.exists():
                _retry(
                    client.retrieve,
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": variables,
                        "year": str(current.year),
                        "month": f"{current.month:02d}",
                        "day": [f"{d:02d}" for d in range(1, days_in_month + 1)],
                        "time": [f"{h:02d}:00" for h in range(24)],
                        "area": [bbox["lat_max"], bbox["lon_min"],
                                 bbox["lat_min"], bbox["lon_max"]],
                        "data_format": "netcdf",
                        "download_format": "unarchived",
                    },
                    str(out_file),
                )

            # CDS-Beta sometimes returns a zip even when "unarchived" is asked.
            # Detect by magic bytes and extract the first NetCDF inside.
            with open(out_file, "rb") as _f:
                _magic = _f.read(4)
            if _magic == b"PK\x03\x04":
                import zipfile
                with zipfile.ZipFile(out_file) as zf:
                    nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
                    if not nc_names:
                        raise RuntimeError(f"No .nc inside zipped CDS response {out_file}")
                    extract_dir = out_file.with_suffix(".extracted")
                    extract_dir.mkdir(exist_ok=True)
                    nc_paths = [extract_dir / n for n in nc_names]
                    for n, p in zip(nc_names, nc_paths):
                        p.write_bytes(zf.read(n))
                import xarray as xr
                if len(nc_paths) == 1:
                    ds = xr.open_dataset(nc_paths[0])
                else:
                    # CDS-Beta splits accumulated vs instant variables across files.
                    # Open each in-memory and merge — no dask needed.
                    ds = xr.merge([xr.open_dataset(p) for p in nc_paths],
                                  compat="override")
            else:
                import xarray as xr
                ds = xr.open_dataset(out_file)

            # Long → short variable name mapping. CDS-Beta NetCDFs use the
            # short codes; older CDS used long names. Handle both.
            short_for = {
                "surface_solar_radiation_downwards": "ssrd",
                "total_cloud_cover":                 "tcc",
                "total_column_water_vapour":         "tcwv",
                "boundary_layer_height":             "blh",
                "surface_pressure":                  "sp",
            }
            time_coord = "valid_time" if "valid_time" in ds.coords else "time"

            for _, sta in stations.iterrows():
                point = ds.sel(
                    latitude=sta["latitude"],
                    longitude=sta["longitude"],
                    method="nearest"
                )
                times = point[time_coord].values
                for t in range(len(times)):
                    rec = {
                        "station":   sta["station"],
                        "timestamp": pd.Timestamp(times[t]),
                    }
                    for var in variables:
                        short = short_for[var]
                        ds_var = short if short in point.data_vars else var
                        val = float(point[ds_var].values[t])
                        # ERA5 ssrd is cumulative J/m² per hour → convert to avg W/m²
                        if var == "surface_solar_radiation_downwards":
                            val = val / 3600.0
                        rec[f"ext_era5_{short}"] = val
                    all_records.append(rec)

            ds.close()
            log.info(f"  {month_str}: extracted {len(stations)} stations")

            # Free disk: drop the per-month .extracted dir now that values are in all_records.
            # The original .nc zip on disk lets us re-extract on resume without re-downloading.
            extracted_dir = out_file.with_suffix(".extracted")
            if extracted_dir.exists():
                import shutil
                shutil.rmtree(extracted_dir, ignore_errors=True)

        except Exception as e:
            log.warning(f"  ERA5 {month_str} failed: {e}")

        # Checkpoint
        if all_records:
            _safe_write_csv(all_records, checkpoint_path)

        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)

    df = pd.DataFrame(all_records)
    if df.empty:
        log.error("No ERA5 data extracted.")
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset=["station", "timestamp"])
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)

    out = SAT_DIR / "era5_hourly.csv"
    df.to_csv(out, index=False)
    log.info(f"Saved: {out}  ({len(df):,} rows)")

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 5: CAMS Aerosol Optical Depth via CDS API
# ═══════════════════════════════════════════════════════════════════════════════

def extract_cams(cds_key: str, ads_url: str | None = None) -> pd.DataFrame:
    """
    Download CAMS global reanalysis aerosol optical depth.

    CAMS reanalysis lives at the Atmosphere Data Store (ADS), NOT the Climate
    Data Store. You need to register at https://ads.atmosphere.copernicus.eu/
    and accept the licence for `cams-global-reanalysis-eac4` once. Your CDS
    PAT usually works at ADS as well; pass it as `cds_key`.

    Output columns:
      ext_cams_aod550       — aerosol optical depth at 550nm
      ext_cams_duaod550     — dust aerosol optical depth at 550nm
      ext_cams_bcaod550     — black carbon AOD at 550nm
    """
    try:
        import cdsapi
    except ImportError:
        log.error("pip install cdsapi")
        sys.exit(1)

    log.info("=== CAMS Aerosol Extraction ===")

    stations = load_stations()
    bbox = _station_bbox(stations, buffer=1.0)

    url = ads_url or "https://ads.atmosphere.copernicus.eu/api"
    log.info(f"Using ADS endpoint: {url}")
    client = cdsapi.Client(url=url, key=cds_key)

    variables = [
        "total_aerosol_optical_depth_550nm",
        "dust_aerosol_optical_depth_550nm",
        "black_carbon_aerosol_optical_depth_550nm",
        "organic_matter_aerosol_optical_depth_550nm",
        "sulphate_aerosol_optical_depth_550nm",
        "sea_salt_aerosol_optical_depth_550nm",
        "total_column_water_vapour",
    ]

    ts_df = load_timestamps()
    date_min = ts_df["timestamp"].dt.date.min()
    date_max = ts_df["timestamp"].dt.date.max()

    checkpoint_path = SAT_DIR / "cams_checkpoint.csv"
    all_records = _safe_read_csv(checkpoint_path)
    all_records = all_records.to_dict("records") if all_records is not None else []

    current = datetime(date_min.year, date_min.month, 1)
    end = datetime(date_max.year, date_max.month, 1)

    while current <= end:
        month_str = current.strftime("%Y-%m")
        log.info(f"  Downloading CAMS {month_str} ...")

        days_in_month = ((current.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)).day
        out_file = SAT_DIR / f"cams_{month_str}.nc"

        # Defensive: drop truncated cached files (CAMS months are ~6 MB normally)
        if out_file.exists() and out_file.stat().st_size < 1_000_000:
            log.warning(f"  cached {out_file.name} suspiciously small "
                        f"({out_file.stat().st_size:,}B), deleting for re-download")
            out_file.unlink()

        try:
            if not out_file.exists():
                _retry(
                    client.retrieve,
                    "cams-global-reanalysis-eac4",
                    {
                        "variable": variables,
                        "date": f"{current:%Y-%m-%d}/{current.replace(day=days_in_month):%Y-%m-%d}",
                        "time": [f"{h:02d}:00" for h in range(0, 24, 3)],
                        "area": [bbox["lat_max"], bbox["lon_min"],
                                 bbox["lat_min"], bbox["lon_max"]],
                        "data_format": "netcdf",
                        "download_format": "unarchived",
                    },
                    str(out_file),
                )

            with open(out_file, "rb") as _f:
                _magic = _f.read(4)
            if _magic == b"PK\x03\x04":
                import zipfile
                with zipfile.ZipFile(out_file) as zf:
                    nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
                    if not nc_names:
                        raise RuntimeError(f"No .nc inside zipped CDS response {out_file}")
                    extract_dir = out_file.with_suffix(".extracted")
                    extract_dir.mkdir(exist_ok=True)
                    nc_paths = [extract_dir / n for n in nc_names]
                    for n, p in zip(nc_names, nc_paths):
                        p.write_bytes(zf.read(n))
                import xarray as xr
                if len(nc_paths) == 1:
                    ds = xr.open_dataset(nc_paths[0])
                else:
                    # CDS-Beta splits accumulated vs instant variables across files.
                    # Open each in-memory and merge — no dask needed.
                    ds = xr.merge([xr.open_dataset(p) for p in nc_paths],
                                  compat="override")
            else:
                import xarray as xr
                ds = xr.open_dataset(out_file)

            short_for = {
                "total_aerosol_optical_depth_550nm":          "aod550",
                "dust_aerosol_optical_depth_550nm":           "duaod550",
                "black_carbon_aerosol_optical_depth_550nm":   "bcaod550",
                "organic_matter_aerosol_optical_depth_550nm": "omaod550",
                "sulphate_aerosol_optical_depth_550nm":       "suaod550",   # ADS uses "suaod" not "su4aod"
                "sea_salt_aerosol_optical_depth_550nm":       "ssaod550",
                "total_column_water_vapour":                  "tcwv",
            }
            time_coord = "valid_time" if "valid_time" in ds.coords else "time"

            for _, sta in stations.iterrows():
                point = ds.sel(
                    latitude=sta["latitude"],
                    longitude=sta["longitude"],
                    method="nearest"
                )
                times = point[time_coord].values
                for t in range(len(times)):
                    rec = {
                        "station":   sta["station"],
                        "timestamp": pd.Timestamp(times[t]),
                    }
                    for var in variables:
                        short = short_for[var]
                        ds_var = short if short in point.data_vars else var
                        rec[f"ext_cams_{short}"] = float(point[ds_var].values[t])
                    all_records.append(rec)

            ds.close()
            log.info(f"  {month_str}: extracted {len(stations)} stations")

            extracted_dir = out_file.with_suffix(".extracted")
            if extracted_dir.exists():
                import shutil
                shutil.rmtree(extracted_dir, ignore_errors=True)

        except Exception as e:
            log.warning(f"  CAMS {month_str} failed: {e}")

        if all_records:
            _safe_write_csv(all_records, checkpoint_path)

        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)

    df = pd.DataFrame(all_records)
    if df.empty:
        log.error("No CAMS data extracted.")
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset=["station", "timestamp"])
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)

    out = SAT_DIR / "cams_aerosol.csv"
    df.to_csv(out, index=False)
    log.info(f"Saved: {out}  ({len(df):,} rows)")

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 6: MODIS (NASA Earthdata — LAADS DAAC)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_modis(earthdata_token: str) -> pd.DataFrame:
    """
    Download MODIS Aqua/Terra daily cloud and albedo products at station locations.
    Uses NASA LAADS DAAC API.

    Products:
      MCD43A3 — MODIS daily surface albedo (500m)
      MOD08_D3 — MODIS daily atmosphere gridded product (1 degree)

    Output columns:
      ext_modis_cloud_fraction    — daily cloud fraction
      ext_modis_cloud_opt_thick   — cloud optical thickness
      ext_modis_aod               — aerosol optical depth
      ext_modis_albedo_wsa        — white-sky albedo (shortwave)
    """
    import requests

    log.info("=== MODIS Extraction (NASA Earthdata) ===")

    stations = load_stations()
    ts_df = load_timestamps()
    date_min = ts_df["timestamp"].dt.date.min()
    date_max = ts_df["timestamp"].dt.date.max()

    headers = {"Authorization": f"Bearer {earthdata_token}"}
    base_url = "https://ladsweb.modaps.eosdis.nasa.gov/api/v2"

    checkpoint_path = SAT_DIR / "modis_checkpoint.csv"
    all_records = _safe_read_csv(checkpoint_path)
    all_records = all_records.to_dict("records") if all_records is not None else []

    # Use MOD08_D3 (daily 1-degree gridded atmosphere product) for cloud/aerosol
    current = datetime(date_min.year, date_min.month, date_min.day)
    end = datetime(date_max.year, date_max.month, date_max.day)
    delta = timedelta(days=1)

    while current <= end:
        doy = current.timetuple().tm_yday
        year = current.year

        try:
            # Search for MOD08_D3 granule for this date
            search_url = (
                f"{base_url}/content/archives/allData/61/MOD08_D3/{year}/{doy:03d}"
            )
            resp = _retry(requests.get, search_url, headers=headers, timeout=30)
            resp.raise_for_status()
            files = resp.json()

            hdf_files = [f for f in files if f.get("name", "").endswith(".hdf")]
            if not hdf_files:
                current += delta
                continue

            # Download the file
            file_name = hdf_files[0]["name"]
            download_url = f"{base_url}/content/archives/allData/61/MOD08_D3/{year}/{doy:03d}/{file_name}"
            data_path = SAT_DIR / f"modis_{current:%Y%m%d}.hdf"

            if not data_path.exists():
                resp = _retry(requests.get, download_url, headers=headers, timeout=120)
                resp.raise_for_status()
                data_path.write_bytes(resp.content)

            # Extract at station locations
            try:
                from pyhdf.SD import SD, SDC
                hdf = SD(str(data_path), SDC.READ)

                # MOD08_D3 is on a 1x1 degree grid (180 lat x 360 lon)
                cloud_frac = hdf.select("Cloud_Fraction_Mean").get()
                cloud_opt  = hdf.select("Cloud_Optical_Thickness_Combined_Mean").get()
                aod        = hdf.select("Aerosol_Optical_Depth_Land_Ocean_Mean").get()

                for _, sta in stations.iterrows():
                    lat_idx = round(90 - sta["latitude"])
                    lon_idx = round(sta["longitude"] + 180)
                    lat_idx = int(np.clip(lat_idx, 0, 179))
                    lon_idx = int(np.clip(lon_idx, 0, 359))

                    rec = {
                        "station": sta["station"],
                        "date": current.date(),
                        "ext_modis_cloud_fraction":  float(cloud_frac[lat_idx, lon_idx]),
                        "ext_modis_cloud_opt_thick": float(cloud_opt[lat_idx, lon_idx]),
                        "ext_modis_aod":             float(aod[lat_idx, lon_idx]),
                    }
                    # Replace fill values
                    for k in ["ext_modis_cloud_fraction", "ext_modis_cloud_opt_thick", "ext_modis_aod"]:
                        if rec[k] < -900 or rec[k] > 10000:
                            rec[k] = np.nan
                    all_records.append(rec)

                hdf.end()
            except ImportError:
                log.error("pip install python-hdf4  (pyhdf)")
                sys.exit(1)

            # Clean up daily file
            if data_path.exists():
                data_path.unlink()

        except Exception as e:
            log.debug(f"  MODIS {current:%Y-%m-%d}: {e}")

        current += delta
        if current.day == 1:
            log.info(f"  MODIS completed {(current - delta):%Y-%m}")
            if all_records:
                _safe_write_csv(all_records, checkpoint_path)

    df = pd.DataFrame(all_records)
    if df.empty:
        log.error("No MODIS data extracted.")
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["station", "date"])
    df = df.sort_values(["station", "date"]).reset_index(drop=True)

    out = SAT_DIR / "modis_daily.csv"
    df.to_csv(out, index=False)
    log.info(f"Saved: {out}  ({len(df):,} rows)")

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 7: MSG Cloud Physical Properties (EUMETSAT)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_msg_cloud(consumer_key: str, consumer_secret: str) -> pd.DataFrame:
    """
    Download MSG SEVIRI Cloud Physical Properties at 15-min resolution.
    Uses the same EUMETSAT Data Store as LSA-SAF.

    Product: EO:EUM:DAT:MSG:CLP — Cloud Physical Properties
    Output columns:
      ext_msg_cloud_type     — cloud type classification (0-20)
      ext_msg_cloud_phase    — thermodynamic phase (water/ice/mixed)
      ext_msg_cloud_opt_thick — cloud optical thickness
    """
    try:
        import eumdac
    except ImportError:
        log.error("pip install eumdac h5py")
        sys.exit(1)
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        log.error("pip install scipy")
        sys.exit(1)

    log.info("=== MSG Cloud Physical Properties Extraction ===")

    token = eumdac.AccessToken((consumer_key, consumer_secret))
    datastore = eumdac.DataStore(token)
    stations = load_stations()

    ts_df = load_timestamps()
    date_min = ts_df["timestamp"].dt.date.min()
    date_max = ts_df["timestamp"].dt.date.max()
    start_dt = datetime(date_min.year, date_min.month, 1)
    end_dt   = datetime(date_max.year, date_max.month, date_max.day, 23, 59)

    collection = datastore.get_collection("EO:EUM:DAT:MSG:CLP")
    log.info(f"Collection: {collection.title}")

    checkpoint_path = SAT_DIR / "msg_cloud_checkpoint.csv"
    existing = _safe_read_csv(checkpoint_path, parse_dates=["timestamp"])
    if existing is not None:
        records = existing.to_dict("records")
        last_ts = existing["timestamp"].max()
        start_dt = pd.Timestamp(last_ts).to_pydatetime().replace(day=1)
        log.info(f"Resuming from {start_dt:%Y-%m}")
    else:
        records = []

    grid_tree = None
    grid_valid_mask = None

    current = start_dt
    while current <= end_dt:
        month_end = (current.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(seconds=1)
        month_end = min(month_end, end_dt)

        log.info(f"  MSG CLP {current:%Y-%m} ...")
        try:
            products = list(_retry(collection.search, dtstart=current, dtend=month_end))
        except Exception as e:
            log.warning(f"  Search failed: {e}")
            current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
            continue

        for product in products:
            try:
                import io, h5py
                with product.open() as fsrc:
                    data_bytes = fsrc.read()

                ts_str = _parse_eumetsat_timestamp(str(product))
                if ts_str is None:
                    continue
                ts = pd.Timestamp(ts_str)

                with h5py.File(io.BytesIO(data_bytes), "r") as f:
                    # Read cloud properties
                    cloud_type = f["CT"][:] if "CT" in f else None
                    cloud_phase = f["CPH"][:] if "CPH" in f else None
                    cloud_cot = f["COT"][:] if "COT" in f else None
                    lat_grid = f["LATITUDE"][:] if "LATITUDE" in f else None
                    lon_grid = f["LONGITUDE"][:] if "LONGITUDE" in f else None

                if lat_grid is None or cloud_type is None:
                    continue

                if grid_tree is None:
                    lat_flat = lat_grid.ravel()
                    lon_flat = lon_grid.ravel()
                    grid_valid_mask = np.isfinite(lat_flat) & np.isfinite(lon_flat)
                    grid_tree = cKDTree(np.column_stack([
                        lat_flat[grid_valid_mask], lon_flat[grid_valid_mask]
                    ]))

                for _, sta in stations.iterrows():
                    dist, idx = grid_tree.query([sta["latitude"], sta["longitude"]], k=1)
                    if dist > 0.25:
                        continue

                    ct_val  = float(cloud_type.ravel()[grid_valid_mask][idx]) if cloud_type is not None else np.nan
                    cph_val = float(cloud_phase.ravel()[grid_valid_mask][idx]) if cloud_phase is not None else np.nan
                    cot_val = float(cloud_cot.ravel()[grid_valid_mask][idx]) if cloud_cot is not None else np.nan

                    # Clean fill values
                    ct_val  = np.nan if (ct_val < 0 or ct_val > 10000) else ct_val
                    cph_val = np.nan if (cph_val < 0 or cph_val > 10000) else cph_val
                    cot_val = np.nan if (cot_val < 0 or cot_val > 10000) else cot_val

                    records.append({
                        "station":                  sta["station"],
                        "timestamp":                ts,
                        "ext_msg_cloud_type":       ct_val,
                        "ext_msg_cloud_phase":      cph_val,
                        "ext_msg_cloud_opt_thick":  cot_val,
                    })
            except Exception as e:
                log.debug(f"  MSG CLP error: {e}")

        if records:
            _safe_write_csv(records, checkpoint_path)

        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)

    df = pd.DataFrame(records)
    if df.empty:
        log.error("No MSG cloud data extracted.")
        return df

    df = df.drop_duplicates(subset=["station", "timestamp"])
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)

    out = SAT_DIR / "msg_cloud.csv"
    df.to_csv(out, index=False)
    log.info(f"Saved: {out}  ({len(df):,} rows)")

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PREPARE: Create clean Train_Test_Merged.csv
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_merged_base() -> Path:
    """
    Combine Train.csv + Test.csv into a single sorted file with a 'split' column.
    This is the base file all external features get merged onto.

    Output: data/processed/Train_Test_Merged.csv
      - 'split' column: 'train' or 'test'
      - timestamps parsed and sorted per station
      - sorted by (station, timestamp)
    """
    log.info("=== Preparing Train_Test_Merged.csv ===")

    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)

    train["split"] = "train"
    test["split"]  = "test"

    df = pd.concat([train, test], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", dayfirst=True)
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out = PROC_DIR / "Train_Test_Merged.csv"
    df.to_csv(out, index=False)

    log.info(f"Saved: {out}  ({df.shape})")
    log.info(f"  Stations: {df['station'].nunique()} | "
             f"Train: {(df['split']=='train').sum():,} | "
             f"Test: {(df['split']=='test').sum():,}")
    log.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MERGE: Attach all external features to Train/Test
# ═══════════════════════════════════════════════════════════════════════════════

def merge_and_save():
    """
    Merge all available satellite/computed CSVs into enhanced Train/Test.
    Skips any source whose CSV doesn't exist yet.
    """
    log.info("=== Merging all external features into Train/Test ===")

    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    train["timestamp"] = pd.to_datetime(train["timestamp"], format="mixed", dayfirst=True)
    test["timestamp"]  = pd.to_datetime(test["timestamp"],  format="mixed", dayfirst=True)
    original_cols = list(train.columns)
    log.info(f"Train: {train.shape} | Test: {test.shape}")

    def _attach_timestamp_source(df, csv_path, merge_cols, label):
        """Merge a source CSV on (station, timestamp) with 15-min rounding."""
        if not csv_path.exists() or csv_path.stat().st_size < 10:
            log.warning(f"  {label}: {csv_path.name} not found or empty, skipping")
            return df
        sat = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df["_ts_rounded"]  = df["timestamp"].dt.round("15min")
        sat["_ts_rounded"] = sat["timestamp"].dt.round("15min")
        ext_cols = [c for c in sat.columns if c.startswith("ext_")]
        df = df.merge(
            sat[["station", "_ts_rounded"] + ext_cols],
            on=["station", "_ts_rounded"], how="left"
        ).drop(columns=["_ts_rounded"])
        filled = df[ext_cols[0]].notna().sum() if ext_cols else 0
        log.info(f"  {label}: {filled:,}/{len(df):,} matched ({filled/len(df)*100:.1f}%)")
        return df

    def _attach_hourly_source(df, csv_path, label):
        """Merge an hourly source CSV on (station, hour) — forward-fill to 15-min."""
        if not csv_path.exists() or csv_path.stat().st_size < 10:
            log.warning(f"  {label}: {csv_path.name} not found or empty, skipping")
            return df
        sat = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df["_ts_hour"]  = df["timestamp"].dt.floor("h")
        sat["_ts_hour"] = sat["timestamp"].dt.floor("h")
        ext_cols = [c for c in sat.columns if c.startswith("ext_")]
        df = df.merge(
            sat[["station", "_ts_hour"] + ext_cols].drop_duplicates(subset=["station", "_ts_hour"]),
            on=["station", "_ts_hour"], how="left"
        ).drop(columns=["_ts_hour"])
        filled = df[ext_cols[0]].notna().sum() if ext_cols else 0
        log.info(f"  {label}: {filled:,}/{len(df):,} matched ({filled/len(df)*100:.1f}%)")
        return df

    def _attach_daily_source(df, csv_path, label):
        """Merge a daily source CSV on (station, date)."""
        if not csv_path.exists() or csv_path.stat().st_size < 10:
            log.warning(f"  {label}: {csv_path.name} not found or empty, skipping")
            return df
        sat = pd.read_csv(csv_path, parse_dates=["date"])
        df["_date"]  = df["timestamp"].dt.date
        sat["_date"] = sat["date"].dt.date
        ext_cols = [c for c in sat.columns if c.startswith("ext_")]
        df = df.merge(
            sat[["station", "_date"] + ext_cols],
            on=["station", "_date"], how="left"
        ).drop(columns=["_date"])
        filled = df[ext_cols[0]].notna().sum() if ext_cols else 0
        log.info(f"  {label}: {filled:,}/{len(df):,} matched ({filled/len(df)*100:.1f}%)")
        return df

    # Define all sources and their merge strategy
    sources = [
        # (csv_path, merge_type, label)
        (SAT_DIR / "solar_features.csv",     "timestamp", "Solar geometry"),
        (SAT_DIR / "pvlib_features.csv",     "timestamp", "pvlib clear-sky"),
        (SAT_DIR / "temporal_neighbors.csv", "timestamp", "Temporal neighbours"),
        (SAT_DIR / "forward_weather.csv",    "timestamp", "Forward weather"),
        (SAT_DIR / "same_day_aggregates.csv","timestamp", "Same-day aggregates"),
        (SAT_DIR / "energy_balance.csv",     "timestamp", "Energy-balance inversion"),
        (SAT_DIR / "cams_radiation_ts.csv",  "timestamp", "CAMS solar radiation timeseries"),
        (SAT_DIR / "sarah_radiation.csv",    "daily",     "SARAH-3 radiation"),
        (SAT_DIR / "lsa_saf_mdssftd.csv",    "daily",     "LSA-SAF MDSSFTD (SW+diffuse-frac)"),
        (SAT_DIR / "lsa_saf_mlst.csv",       "daily",     "LSA-SAF MLST"),
        (SAT_DIR / "lsa_saf_mdslf.csv",      "daily",     "LSA-SAF MDSLF (longwave)"),
        # MSG Cloud: EO:EUM:DAT:MSG:CLP removed from EUMETSAT Data Store
        # (SAT_DIR / "msg_cloud.csv",        "timestamp", "MSG Cloud"),
        (SAT_DIR / "era5_hourly.csv",        "hourly",    "ERA5"),
        (SAT_DIR / "open_meteo_hourly.csv",  "hourly",    "Open-Meteo"),
        (SAT_DIR / "nasa_power_hourly.csv",  "hourly",    "NASA POWER"),
        (SAT_DIR / "cams_aerosol.csv",       "hourly",    "CAMS"),
        (SAT_DIR / "merra2_aerosols.csv",    "hourly",    "MERRA-2 aerosols"),
        (SAT_DIR / "tropomi_cloud.csv",      "daily",     "TROPOMI cloud"),
        (SAT_DIR / "tropomi_aerosol.csv",    "daily",     "TROPOMI aerosol"),
        (SAT_DIR / "modis_daily.csv",        "daily",     "MODIS"),
    ]

    for csv_path, merge_type, label in sources:
        if merge_type == "timestamp":
            train = _attach_timestamp_source(train, csv_path, [], label)
            test  = _attach_timestamp_source(test,  csv_path, [], label)
        elif merge_type == "hourly":
            train = _attach_hourly_source(train, csv_path, label)
            test  = _attach_hourly_source(test,  csv_path, label)
        elif merge_type == "daily":
            train = _attach_daily_source(train, csv_path, label)
            test  = _attach_daily_source(test,  csv_path, label)

    new_cols = [c for c in train.columns if c not in original_cols]
    log.info(f"\nNew columns added ({len(new_cols)}):")
    for col in new_cols:
        tr_pct = train[col].notna().mean() * 100
        te_pct = test[col].notna().mean() * 100
        log.info(f"  {col}: train {tr_pct:.1f}%  test {te_pct:.1f}%")

    train_out = PROC_DIR / "Train_enhanced.csv"
    test_out  = PROC_DIR / "Test_enhanced.csv"
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_out, index=False)
    test.to_csv(test_out, index=False)
    log.info(f"\nSaved: {train_out}  ({train.shape})")
    log.info(f"Saved: {test_out}   ({test.shape})")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TAHMO external data extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
All external columns are prefixed:  ext_sol_, ext_lsa_, ext_tro_,
ext_era5_, ext_cams_, ext_modis_, ext_msg_

Drop all external:  df[[c for c in df.columns if not c.startswith('ext_')]]
Drop one source:    df.drop(columns=[c for c in df.columns if c.startswith('ext_era5_')])

Examples:
  python data_pipeline.py --source solar                          # no API needed
  python data_pipeline.py --source lsa_saf --eumetsat-key K --eumetsat-secret S
  python data_pipeline.py --source era5 --cds-key UID:API_KEY
  python data_pipeline.py --source all ...                        # everything
  python data_pipeline.py --merge-only                            # just merge CSVs

Account registration (all free):
  EUMETSAT:    https://eoportal.eumetsat.int/
  Copernicus:  https://dataspace.copernicus.eu/
  CDS:         https://cds.climate.copernicus.eu/
  NASA:        https://urs.earthdata.nasa.gov/
        """,
    )
    parser.add_argument(
        "--source", choices=ALL_SOURCES + ["all"],
        default="all", help="Which source to download (default: all)"
    )
    parser.add_argument("--eumetsat-key",    default="", help="EUMETSAT consumer key")
    parser.add_argument("--eumetsat-secret", default="", help="EUMETSAT consumer secret")
    parser.add_argument("--cdse-user",       default="", help="Copernicus Data Space username")
    parser.add_argument("--cdse-password",   default="", help="Copernicus Data Space password")
    parser.add_argument("--cds-key",         default="", help="CDS API key (format: UID:KEY)")
    parser.add_argument("--earthdata-token", default="", help="NASA Earthdata bearer token")
    parser.add_argument("--merge-only",      action="store_true", help="Skip downloads, just merge")

    args = parser.parse_args()

    # Always create the clean merged base first
    prepare_merged_base()

    if args.merge_only:
        merge_and_save()
        return

    sources = ALL_SOURCES if args.source == "all" else [args.source]

    # Solar — no credentials needed
    if "solar" in sources:
        compute_solar_features()

    # LSA-SAF — EUMETSAT
    if "lsa_saf" in sources:
        if not args.eumetsat_key or not args.eumetsat_secret:
            log.error("--eumetsat-key and --eumetsat-secret required for LSA-SAF")
            sys.exit(1)
        extract_lsa_saf(args.eumetsat_key, args.eumetsat_secret)

    # MSG Cloud — disabled: EO:EUM:DAT:MSG:CLP removed from EUMETSAT Data Store
    if "msg_cloud" in sources:
        log.warning("MSG Cloud (CLP) skipped — collection removed from EUMETSAT Data Store")

    # TROPOMI — Copernicus Data Space
    if "tropomi" in sources:
        if not args.cdse_user or not args.cdse_password:
            log.error("--cdse-user and --cdse-password required for TROPOMI")
            sys.exit(1)
        extract_tropomi(args.cdse_user, args.cdse_password)

    # ERA5 — CDS
    if "era5" in sources:
        if not args.cds_key:
            log.error("--cds-key required for ERA5")
            sys.exit(1)
        extract_era5(args.cds_key)

    # CAMS — CDS (same credentials)
    if "cams" in sources:
        if not args.cds_key:
            log.error("--cds-key required for CAMS")
            sys.exit(1)
        extract_cams(args.cds_key)

    # MODIS — NASA Earthdata
    if "modis" in sources:
        if not args.earthdata_token:
            log.error("--earthdata-token required for MODIS")
            sys.exit(1)
        extract_modis(args.earthdata_token)

    # Merge everything
    merge_and_save()


if __name__ == "__main__":
    main()
