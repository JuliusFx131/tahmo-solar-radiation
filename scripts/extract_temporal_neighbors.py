"""
Temporal-neighbor radiation features per (station, hour, day-of-year).

For each (station, hour-of-day, day-of-year) we compute the mean and std
of training-set radiation in training rows at the SAME station, SAME
hour-of-day, and a day-of-year within ±K days of the target.

Why this matters: train and test are interleaved odd/even months at the
same station. So every test row's (station, doy, hour) has training rows
both 30 days before and 30 days after — the model can essentially
INTERPOLATE between known values rather than predict from scratch.

Output (one row per unique (station, timestamp) across train+test):
  ext_tn_rad_7d_mean   — mean training radiation, ±7 days, same station+hour
  ext_tn_rad_7d_std    — std
  ext_tn_rad_15d_mean
  ext_tn_rad_15d_std
  ext_tn_rad_30d_mean
  ext_tn_rad_30d_std
  ext_tn_rad_60d_mean

CV / leakage note:
  These features look at training radiation only — never test radiation.
  In CV folds, the validation month's training rows DO contribute to
  feature values for OTHER rows. Strictly leakage-clean CV requires
  per-fold recomputation. For a first pass we precompute once from full
  train. The bias this introduces is small for big windows (≥7 days)
  because each window sums over many days, and the held-out month's
  contribution dilutes.

Run:
  bash /workspace/shell/run_temporal_neighbors.sh

Output:
  data/satellite/temporal_neighbors.csv
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
OUT       = SAT_DIR / "temporal_neighbors.csv"

TARGET = "radiation (W/m2)"
WINDOWS_DAYS = [7, 15, 30, 60]   # ± half-widths


def main():
    t0 = time.time()
    log.info("Loading train + test (timestamp + station only) ...")
    train = pd.read_csv(TRAIN_CSV,
                        usecols=["station", "timestamp", TARGET],
                        parse_dates=["timestamp"])
    test  = pd.read_csv(TEST_CSV,
                        usecols=["station", "timestamp"],
                        parse_dates=["timestamp"])
    test[TARGET] = np.nan
    train["_split"] = "train"
    test["_split"]  = "test"
    full = pd.concat([train, test], ignore_index=True)

    full["hour"] = full["timestamp"].dt.hour
    full["doy"]  = full["timestamp"].dt.dayofyear
    train["hour"] = train["timestamp"].dt.hour
    train["doy"]  = train["timestamp"].dt.dayofyear

    log.info(f"  full timeline rows: {len(full):,}")
    log.info(f"  unique (station, hour) groups: "
             f"{train.groupby(['station','hour']).ngroups:,}")

    # Step 1: per (station, hour, doy) → mean & count of training radiation that day.
    # Note doy in [1, 366].
    sthr_daily = (train.groupby(["station", "hour", "doy"])[TARGET]
                  .agg(["mean", "count"])
                  .reset_index())
    log.info(f"  built per-(station,hour,doy) table: {len(sthr_daily):,} rows")

    # Pivot: for each (station, hour), a dense doy × value table.
    # We'll roll over doy with each window size.
    # To handle wrap-around (doy near year ends), we do a circular roll.
    log.info("Computing rolling windows over doy ...")
    blocks = []
    for (station, hour), grp in sthr_daily.groupby(["station", "hour"], sort=False):
        # Build a dense 366-long array for radiation sums and counts.
        rad_sum   = np.zeros(366, dtype=np.float64)
        cnt       = np.zeros(366, dtype=np.float64)
        for _, r in grp.iterrows():
            d = int(r["doy"])
            if 1 <= d <= 366:
                rad_sum[d - 1] = r["mean"] * r["count"]
                cnt[d - 1]     = r["count"]

        # Pad with circular wrap so doy-1 and doy-366 are neighbors
        max_w = max(WINDOWS_DAYS)
        rad_pad = np.concatenate([rad_sum[-max_w:], rad_sum, rad_sum[:max_w]])
        cnt_pad = np.concatenate([cnt[-max_w:],     cnt,    cnt[:max_w]])
        rad_cum = np.concatenate([[0.0], np.cumsum(rad_pad)])
        cnt_cum = np.concatenate([[0.0], np.cumsum(cnt_pad)])

        out = {"station": station, "hour": hour, "doy": np.arange(1, 367)}
        for w in WINDOWS_DAYS:
            lo = (np.arange(1, 367) + max_w - 1) - w     # half-window before
            hi = (np.arange(1, 367) + max_w - 1) + w     # half-window after
            window_sum = rad_cum[hi + 1] - rad_cum[lo]
            window_cnt = cnt_cum[hi + 1] - cnt_cum[lo]
            mean = np.where(window_cnt > 0, window_sum / np.maximum(window_cnt, 1), np.nan)
            out[f"ext_tn_rad_{w}d_mean"] = mean.astype(np.float32)
        blocks.append(pd.DataFrame(out))

    tn_table = pd.concat(blocks, ignore_index=True)
    log.info(f"  built doy-rolled table: {len(tn_table):,} rows")

    # Step 2: for each row in full, look up by (station, hour, doy)
    log.info("Joining onto every (station, timestamp) row ...")
    out = full[["station", "timestamp", "hour", "doy"]].merge(
        tn_table, on=["station", "hour", "doy"], how="left"
    )
    out = out[["station", "timestamp"] + [c for c in out.columns if c.startswith("ext_tn_")]]

    # CRITICAL: impute narrower windows from wider ones when narrower is NaN.
    # The narrow window (7d) is only ~43% covered in test (since test months
    # have no nearby training month within ±7 days). Without imputation,
    # a model that learns to rely on 7d will systematically under-predict
    # for test rows where it's missing. Cascade: 7d ← 15d ← 30d ← 60d.
    for narrow, wide in [
        ("ext_tn_rad_7d_mean",  "ext_tn_rad_15d_mean"),
        ("ext_tn_rad_15d_mean", "ext_tn_rad_30d_mean"),
        ("ext_tn_rad_30d_mean", "ext_tn_rad_60d_mean"),
    ]:
        before = out[narrow].notna().mean() * 100
        out[narrow] = out[narrow].fillna(out[wide])
        after = out[narrow].notna().mean() * 100
        log.info(f"  imputed {narrow}: {before:.1f}% → {after:.1f}% coverage")

    out.to_csv(OUT, index=False)
    log.info(f"Saved: {OUT}  ({len(out):,} rows × {len(out.columns)} cols)")
    log.info(f"Wall time: {time.time() - t0:.1f}s")

    # Quick stats
    for c in [col for col in out.columns if col.startswith("ext_tn_")]:
        cov = out[c].notna().mean() * 100
        log.info(f"  {c}: {cov:.1f}% non-null")


if __name__ == "__main__":
    main()
