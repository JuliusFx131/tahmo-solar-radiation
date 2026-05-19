"""
Same-day weather aggregates per (station, date).

For every test row at (station, date=D, hour=H), we already have weather at
all 96 15-min slots of that same day (train+test together cover whole days).
The day's character — sunny vs. overcast — is summarised cheaply by daily
statistics. We compute them once and join to every row of that day at that
station.

Output columns (ext_dd_* prefix; "dd" = daily-day):
  Per (station, date) and propagated to all 96 rows:

  ext_dd_temp_max, ext_dd_temp_min, ext_dd_temp_amp (max-min)
  ext_dd_temp_mean, ext_dd_temp_std
  ext_dd_rh_max, ext_dd_rh_min, ext_dd_rh_mean
  ext_dd_precip_sum, ext_dd_precip_max
  ext_dd_om_ghi_max, ext_dd_om_ghi_sum
  ext_dd_om_cc_mean  (mean cloud cover today)
  ext_dd_np_ghi_max, ext_dd_np_ghi_sum
  ext_dd_ssrd_max,   ext_dd_ssrd_sum  (ERA5)

All 100% covered (every row has its own day's aggregates).

Run:
  bash /workspace/shell/run_same_day_aggregates.sh

Output:
  data/satellite/same_day_aggregates.csv
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent
PROC     = ROOT / "data" / "processed"
SAT_DIR  = ROOT / "data" / "satellite"
SAT_DIR.mkdir(parents=True, exist_ok=True)

ENH_TRAIN = PROC / "Train_enhanced.csv"
ENH_TEST  = PROC / "Test_enhanced.csv"
OUT       = SAT_DIR / "same_day_aggregates.csv"


def main():
    t0 = time.time()
    log.info("Loading enhanced files (only the cols we aggregate over) ...")
    cols_needed = [
        "station", "timestamp",
        "temperature (degrees Celsius)", "relativehumidity (-)", "precipitation (mm)",
        "ext_om_ghi", "ext_om_cc_total",
        "ext_np_allsky_ghi",
        "ext_era5_ssrd",
    ]
    train = pd.read_csv(ENH_TRAIN, usecols=cols_needed, parse_dates=["timestamp"])
    test  = pd.read_csv(ENH_TEST,  usecols=cols_needed, parse_dates=["timestamp"])
    full = pd.concat([train, test], ignore_index=True)
    full["date"] = full["timestamp"].dt.date
    log.info(f"  full rows: {len(full):,},  unique (station, date): "
             f"{full.groupby(['station','date']).ngroups:,}")

    log.info("Aggregating per (station, date) ...")
    agg = full.groupby(["station", "date"], sort=False).agg(
        ext_dd_temp_max  =("temperature (degrees Celsius)", "max"),
        ext_dd_temp_min  =("temperature (degrees Celsius)", "min"),
        ext_dd_temp_mean =("temperature (degrees Celsius)", "mean"),
        ext_dd_temp_std  =("temperature (degrees Celsius)", "std"),
        ext_dd_rh_max    =("relativehumidity (-)",          "max"),
        ext_dd_rh_min    =("relativehumidity (-)",          "min"),
        ext_dd_rh_mean   =("relativehumidity (-)",          "mean"),
        ext_dd_precip_sum=("precipitation (mm)",            "sum"),
        ext_dd_precip_max=("precipitation (mm)",            "max"),
        ext_dd_om_ghi_max=("ext_om_ghi",                    "max"),
        ext_dd_om_ghi_sum=("ext_om_ghi",                    "sum"),
        ext_dd_om_cc_mean=("ext_om_cc_total",               "mean"),
        ext_dd_np_ghi_max=("ext_np_allsky_ghi",             "max"),
        ext_dd_np_ghi_sum=("ext_np_allsky_ghi",             "sum"),
        ext_dd_ssrd_max  =("ext_era5_ssrd",                 "max"),
        ext_dd_ssrd_sum  =("ext_era5_ssrd",                 "sum"),
    ).reset_index()
    agg["ext_dd_temp_amp"] = agg["ext_dd_temp_max"] - agg["ext_dd_temp_min"]

    # Cast to float32
    float_cols = [c for c in agg.columns if c.startswith("ext_dd_")]
    for c in float_cols:
        agg[c] = agg[c].astype(np.float32, copy=False)

    log.info(f"  built daily-aggregate table: {len(agg):,} rows × {len(float_cols)} stats")

    # Join back to every (station, timestamp) row — produce timestamp-keyed table
    log.info("Propagating to every (station, timestamp) row ...")
    out = full[["station", "timestamp", "date"]].merge(
        agg, on=["station", "date"], how="left"
    )
    out = out[["station", "timestamp"] + float_cols]
    out.to_csv(OUT, index=False)
    log.info(f"Saved: {OUT}  ({len(out):,} rows × {len(out.columns)} cols)")
    log.info(f"Wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
