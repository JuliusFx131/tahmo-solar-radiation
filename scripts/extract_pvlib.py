"""
pvlib clear-sky + solar geometry features for every (station, timestamp).

Pure local compute — no API, no auth, no download. Produces a higher-fidelity
clear-sky reference than the simple Ineichen-Perez we already had in
`ext_sol_clearsky`. The competitor (`fgbfgb` on the LB) reported these are
useful, and pvlib has well-tested implementations.

Output columns (ext_pv_* prefix):
  ext_pv_apparent_zenith      — atmosphere-refraction-corrected zenith (deg)
  ext_pv_apparent_elevation   — atmosphere-refraction-corrected elevation (deg)
  ext_pv_airmass_relative     — Kasten-Young airmass
  ext_pv_airmass_absolute     — relative × pressure/1013.25 (altitude-corrected)
  ext_pv_etr                  — extraterrestrial radiation (W/m²)
  ext_pv_linke_turbidity      — monthly climatology from pvlib's lookup
  ext_pv_clearsky_ghi         — Ineichen-Perez clear-sky GHI (W/m²)
  ext_pv_clearsky_dni         — Ineichen-Perez clear-sky DNI (W/m²)
  ext_pv_clearsky_dhi         — Ineichen-Perez clear-sky DHI (W/m²)
  ext_pv_clearsky_ghi_haur    — Haurwitz simple-model clear-sky GHI (W/m²)

Run:
  bash /workspace/shell/run_pvlib.sh

Output:
  data/satellite/pvlib_features.csv
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT / "data" / "raw"
SAT_DIR  = ROOT / "data" / "satellite"
SAT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = RAW_DIR / "Train.csv"
TEST_CSV  = RAW_DIR / "Test.csv"
OUT       = SAT_DIR / "pvlib_features.csv"


def load_stations() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_CSV)
    return (train.groupby("station")[["latitude", "longitude", "elevation"]]
            .first().reset_index())


def load_timestamps() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_CSV, usecols=["station", "timestamp"])
    test  = pd.read_csv(TEST_CSV,  usecols=["station", "timestamp"])
    combined = pd.concat([train, test], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"],
                                           format="mixed", dayfirst=True, utc=True)
    return combined.drop_duplicates(subset=["station", "timestamp"]).reset_index(drop=True)


def main():
    t0 = time.time()
    log.info("Loading stations + timestamps ...")
    stations = load_stations()
    ts_df    = load_timestamps()
    merged   = ts_df.merge(stations, on="station", how="left")

    log.info(f"  {len(stations)} stations, {len(merged):,} (station, timestamp) rows")

    out_chunks = []
    for sta_id, sta_block in merged.groupby("station", sort=False):
        sta_meta = stations[stations["station"] == sta_id].iloc[0]
        lat, lon, elev = sta_meta["latitude"], sta_meta["longitude"], sta_meta["elevation"]
        loc = pvlib.location.Location(lat, lon, altitude=elev, tz="UTC")

        idx = pd.DatetimeIndex(sta_block["timestamp"])
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")

        # Solar position (apparent zenith corrects for atmospheric refraction)
        sp = loc.get_solarposition(idx)
        apparent_zenith    = sp["apparent_zenith"].values
        apparent_elevation = sp["apparent_elevation"].values

        # Airmass (relative + altitude-corrected absolute)
        am_rel = pvlib.atmosphere.get_relative_airmass(apparent_zenith).astype(float)
        # pvlib >=0.10 expects pressure in Pa
        pressure_pa = pvlib.atmosphere.alt2pres(elev)
        am_abs = pvlib.atmosphere.get_absolute_airmass(am_rel, pressure_pa)

        # Extraterrestrial radiation
        etr = pvlib.irradiance.get_extra_radiation(idx).values

        # Linke turbidity (monthly climatology built into pvlib data files)
        try:
            linke = pvlib.clearsky.lookup_linke_turbidity(idx, lat, lon).values
        except Exception as e:
            log.warning(f"  {sta_id}: Linke lookup failed ({e}), defaulting to 3.0")
            linke = np.full(len(idx), 3.0, dtype=float)

        # Ineichen-Perez clear-sky (returns ghi/dni/dhi DataFrame)
        cs = loc.get_clearsky(idx, model="ineichen", linke_turbidity=pd.Series(linke, index=idx))
        cs_ghi = cs["ghi"].values
        cs_dni = cs["dni"].values
        cs_dhi = cs["dhi"].values

        # Haurwitz model (simpler, only needs apparent zenith)
        cs_haur = pvlib.clearsky.haurwitz(pd.Series(apparent_zenith, index=idx)).values.ravel()

        out_chunks.append(pd.DataFrame({
            "station":                    sta_id,
            "timestamp":                  sta_block["timestamp"].values,
            "ext_pv_apparent_zenith":     np.round(apparent_zenith, 3),
            "ext_pv_apparent_elevation":  np.round(apparent_elevation, 3),
            "ext_pv_airmass_relative":    np.round(am_rel, 4),
            "ext_pv_airmass_absolute":    np.round(am_abs, 4),
            "ext_pv_etr":                 np.round(etr, 2),
            "ext_pv_linke_turbidity":     np.round(linke, 3),
            "ext_pv_clearsky_ghi":        np.round(cs_ghi, 2),
            "ext_pv_clearsky_dni":        np.round(cs_dni, 2),
            "ext_pv_clearsky_dhi":        np.round(cs_dhi, 2),
            "ext_pv_clearsky_ghi_haur":   np.round(cs_haur, 2),
        }))
        log.info(f"  {sta_id} done ({len(sta_block):,} rows)")

    out_df = pd.concat(out_chunks, ignore_index=True)
    out_df["timestamp"] = pd.to_datetime(out_df["timestamp"]).dt.tz_localize(None)

    out_df.to_csv(OUT, index=False)
    log.info(f"Saved: {OUT}  ({len(out_df):,} rows × {len(out_df.columns)} cols)")
    log.info(f"Wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
