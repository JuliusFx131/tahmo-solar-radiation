"""
Forward-weather features per (station, timestamp).

The TAHMO competition is an INTERPOLATION task: test rows are missing months
in a continuous timeline. For every test row at (station, T), we have the
station's weather (temperature, humidity, precipitation) for ALL surrounding
timestamps in the same test month — INCLUDING T+15min, T+1h, T+3h, etc.

We've been using only BACKWARD lags. This script adds FORWARD lags. The
intuition: temperature 30 minutes AFTER noon is partly driven by radiation
AT noon. So `temp_lead_30min` carries a (noisy) measurement of current
radiation.

The MOST informative engineered features are the DIFFERENCES:
  • temp_diff_lead_1h = T_at_t+1h - T_at_t
    → "how much did temperature rise in the next hour" — direct proxy for
       energy absorbed by the surface during that hour.
  • rh_diff_lead_1h    → drying as radiation heats the air
  • precip_sum_lead_3h → cloud / rain in the near future

This is not data leakage — the test row's own weather covariates are valid
features. We never look at FUTURE RADIATION (the target).

Output columns (ext_fw_* prefix):
  ext_fw_temp_lead_15m, _30m, _1h, _3h
  ext_fw_temp_diff_1h, _3h
  ext_fw_rh_lead_15m, _30m, _1h, _3h
  ext_fw_rh_diff_1h
  ext_fw_precip_lead_15m, _30m, _1h, _3h
  ext_fw_precip_sum_lead_3h    — cumulative rain in next 3h
  ext_fw_precip_sum_lead_24h   — cumulative rain in next 24h

Run:
  bash /workspace/shell/run_forward_weather.sh

Output:
  data/satellite/forward_weather.csv
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT / "data" / "raw"
SAT_DIR  = ROOT / "data" / "satellite"
SAT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = RAW_DIR / "Train.csv"
TEST_CSV  = RAW_DIR / "Test.csv"
OUT       = SAT_DIR / "forward_weather.csv"

# 15-min cadence → 1 row = 15 min, 2 = 30 min, 4 = 1 h, 12 = 3 h, 96 = 24 h
LEADS = {"15m": 1, "30m": 2, "1h": 4, "3h": 12}


def main():
    t0 = time.time()
    log.info("Loading train + test ...")
    tr = pd.read_csv(TRAIN_CSV,
                     usecols=["station", "timestamp",
                              "temperature (degrees Celsius)",
                              "relativehumidity (-)",
                              "precipitation (mm)"],
                     parse_dates=["timestamp"])
    te = pd.read_csv(TEST_CSV,
                     usecols=["station", "timestamp",
                              "temperature (degrees Celsius)",
                              "relativehumidity (-)",
                              "precipitation (mm)"],
                     parse_dates=["timestamp"])
    full = pd.concat([tr, te], ignore_index=True)
    full = full.sort_values(["station", "timestamp"]).reset_index(drop=True)
    log.info(f"  full rows: {len(full):,}")

    # Compact aliases
    col_temp   = "temperature (degrees Celsius)"
    col_rh     = "relativehumidity (-)"
    col_precip = "precipitation (mm)"

    # Group once
    g = full.groupby("station", sort=False, group_keys=False)

    def _assign(name, arr):
        full[name] = arr.astype(np.float32, copy=False)

    log.info("Computing forward lags ...")
    for label, n in LEADS.items():
        _assign(f"ext_fw_temp_lead_{label}",   g[col_temp].shift(-n).to_numpy())
        _assign(f"ext_fw_rh_lead_{label}",     g[col_rh].shift(-n).to_numpy())
        _assign(f"ext_fw_precip_lead_{label}", g[col_precip].shift(-n).to_numpy())

    log.info("Computing forward differences (change over next 1h / 3h) ...")
    cur_temp = full[col_temp].values.astype(np.float32)
    cur_rh   = full[col_rh].values.astype(np.float32)
    _assign("ext_fw_temp_diff_1h", full["ext_fw_temp_lead_1h"].values - cur_temp)
    _assign("ext_fw_temp_diff_3h", full["ext_fw_temp_lead_3h"].values - cur_temp)
    _assign("ext_fw_rh_diff_1h",   full["ext_fw_rh_lead_1h"].values - cur_rh)

    log.info("Computing forward precip cumulative sums ...")
    # Rolling backwards-from-future-end sum: lead window sum is equivalent to
    # shifting then rolling. Use groupby + rolling on shifted-back series.
    # Easier: for each station, sum precip in next N rows.
    for label, n_rows in [("3h", 12), ("24h", 96)]:
        out = np.empty(len(full), dtype=np.float32)
        for sta, idx in g.indices.items():
            arr = full[col_precip].values[idx].astype(np.float64)
            # forward sum: reverse, cumulative sum window, reverse back
            rev = arr[::-1]
            csum = pd.Series(rev).rolling(n_rows, min_periods=1).sum().to_numpy()
            out[idx] = csum[::-1].astype(np.float32)
        _assign(f"ext_fw_precip_sum_lead_{label}", out)

    fw_cols = [c for c in full.columns if c.startswith("ext_fw_")]
    out_df = full[["station", "timestamp"] + fw_cols]
    out_df.to_csv(OUT, index=False)

    log.info(f"Saved: {OUT}  ({len(out_df):,} rows × {len(out_df.columns)} cols)")
    for c in fw_cols:
        cov = out_df[c].notna().mean() * 100
        log.info(f"  {c}: {cov:.1f}% non-null")
    log.info(f"Wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
