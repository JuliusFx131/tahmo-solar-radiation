"""
Bake every submission file we'd want for tomorrow, given the per-station
metric. All submissions assume LB formula is:
    final = 0.5 × mean_s(|MBE_s|) + 0.5 × mean_s(RMSE_s)

Files produced (ordered by expected payoff, highest first):

  tomorrow_1_n150_pscal.csv
      TargetMBE  = n150 + per-station-correction (from v10 OOF, daytime only)
      TargetRMSE = n150 raw
      Hypothesis: per-station MBE calibration drives |MBE_s| → 0 in expectation,
      pulling the |MBE| half of the score from ~2.7 toward ~0.5-1.0. RMSE half
      stays at ~62.45 (n150's). Expected composite ~31.0-31.7.

  tomorrow_2_v17pscal_n150.csv
      TargetMBE  = v17_wsky + per-station-correction (from v17 OOF, daytime only)
      TargetRMSE = n150 raw
      Hypothesis: v17's loss-weighted training may yield smaller per-station
      bias variance than v10/v12 (OOF |MBE|=1.28 globally suggested this).
      Combined with calibration, this could undercut tomorrow_1's MBE further.

  tomorrow_3_v17_n150_split.csv
      TargetMBE  = v17_wsky raw
      TargetRMSE = n150 raw
      Hypothesis: a raw split, no calibration. Pure test of "best MBE model
      in TargetMBE, best RMSE model in TargetRMSE". Useful baseline.

  tomorrow_4_v17_standalone.csv
      TargetMBE  = v17_wsky
      TargetRMSE = v17_wsky
      Same predictions both columns. Establishes v17's per-station LB stats.

Run:
  $PY scripts/bake_tomorrow_submissions.py
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
SUBS = ROOT / "submissions"

TRAIN = PROC / "Train_enhanced.csv"
TEST  = PROC / "Test_enhanced.csv"

# Source files
N150_PATH   = SUBS / "lgbm_v12_shift_n150.csv"
V17_PATH    = SUBS / "lgbm_v17_wsky.csv"

# OOF prediction sources for per-station calibration
V10_OOF     = SUBS / "lgbm_v10_csr_oof.csv"   # used to calibrate n150
V17_OOF     = SUBS / "lgbm_v17_wsky_oof.csv"  # used to calibrate v17

TARGET = "radiation (W/m2)"


def compute_per_station_bias(oof_path: Path, train_path: Path, label: str) -> pd.Series:
    """For each station, compute mean(y_true - y_pred) on DAYTIME OOF rows
    (ext_sol_elevation > 0). This is the correction to add to daytime
    predictions for that station to drive expected per-station MBE → 0."""
    log.info(f"[{label}] Loading OOF preds from {oof_path.name}")
    oof = pd.read_csv(oof_path, usecols=["ID", "TargetMBE"])
    oof = oof.rename(columns={"TargetMBE": "pred"})

    log.info(f"[{label}] Loading train (ID, station, elev, target)")
    tr = pd.read_csv(train_path,
                     usecols=["ID", "station", "ext_sol_elevation", TARGET])

    j = oof.merge(tr, on="ID", how="inner")
    assert len(j) == len(oof), f"OOF/train ID merge dropped rows: {len(j)} vs {len(oof)}"

    # Daytime only: night rows are already overridden in the submission and
    # don't pass through model predictions in the way OOF does.
    day = j[j["ext_sol_elevation"] > 0]
    log.info(f"[{label}]   daytime OOF rows: {len(day):,} of {len(j):,}")

    bias = day.groupby("station").apply(
        lambda d: (d[TARGET] - d["pred"]).mean()
    ).rename("correction")
    log.info(f"[{label}]   per-station correction stats:")
    log.info(f"[{label}]     n stations = {len(bias)}")
    log.info(f"[{label}]     mean       = {bias.mean():+.3f}")
    log.info(f"[{label}]     std        = {bias.std():.3f}")
    log.info(f"[{label}]     min, max   = {bias.min():+.3f}, {bias.max():+.3f}")
    return bias


def apply_calibration(submission: pd.DataFrame, test_meta: pd.DataFrame,
                      corrections: pd.Series, col: str = "TargetMBE") -> pd.DataFrame:
    """Add the per-station correction to `col` for daytime test rows only.
    Night rows pass through unchanged (they were already per-(station, hour)
    overrides in the source submission)."""
    out = submission.merge(test_meta[["ID", "station", "ext_sol_elevation"]],
                           on="ID", how="left", suffixes=("", ""))
    is_day = out["ext_sol_elevation"] > 0
    delta = out["station"].map(corrections).fillna(0.0).astype(float)
    out[col] = np.clip(out[col].astype(float) + np.where(is_day, delta, 0.0),
                       0.0, 1361.0)
    n_day = int(is_day.sum())
    n_with_corr = int((is_day & out["station"].isin(corrections.index)).sum())
    log.info(f"   applied correction to {n_with_corr:,} / {n_day:,} daytime rows "
             f"({n_with_corr/max(n_day,1)*100:.1f}%)")
    return out[["ID", "TargetMBE", "TargetRMSE"]]


def main():
    log.info("=" * 70)
    log.info("Step 1: per-station bias from v10 OOF (calibration source for n150)")
    log.info("=" * 70)
    corr_v10 = compute_per_station_bias(V10_OOF, TRAIN, "v10_OOF")

    log.info("=" * 70)
    log.info("Step 2: per-station bias from v17 OOF (calibration source for v17)")
    log.info("=" * 70)
    corr_v17 = compute_per_station_bias(V17_OOF, TRAIN, "v17_OOF")

    log.info("=" * 70)
    log.info("Step 3: load test metadata (for station lookup) + source submissions")
    log.info("=" * 70)
    test_meta = pd.read_csv(TEST, usecols=["ID", "station", "ext_sol_elevation"])
    n150 = pd.read_csv(N150_PATH)
    v17  = pd.read_csv(V17_PATH)
    log.info(f"   test_meta: {len(test_meta):,}  n150: {len(n150):,}  v17: {len(v17):,}")

    # ── tomorrow_1: calibrated-n150 / n150 ───────────────────────────────────
    log.info("=" * 70)
    log.info("Bake 1: tomorrow_1_n150_pscal.csv")
    log.info("        TargetMBE = n150 + v10-OOF per-station correction (day)")
    log.info("        TargetRMSE = n150 raw")
    log.info("=" * 70)
    cal_n150 = apply_calibration(n150.copy(), test_meta, corr_v10, col="TargetMBE")
    cal_n150["TargetRMSE"] = n150["TargetRMSE"].astype(float)  # unchanged
    out1 = SUBS / "tomorrow_1_n150_pscal.csv"
    cal_n150.to_csv(out1, index=False)
    log.info(f"   wrote {out1}")
    log.info(f"   TargetMBE mean: {cal_n150['TargetMBE'].mean():.3f}  "
             f"(n150 raw was {n150['TargetMBE'].mean():.3f})")

    # ── tomorrow_2: calibrated-v17 / n150 ───────────────────────────────────
    log.info("=" * 70)
    log.info("Bake 2: tomorrow_2_v17pscal_n150.csv")
    log.info("        TargetMBE = v17 + v17-OOF per-station correction (day)")
    log.info("        TargetRMSE = n150 raw")
    log.info("=" * 70)
    cal_v17 = apply_calibration(v17.copy(), test_meta, corr_v17, col="TargetMBE")
    cal_v17["TargetRMSE"] = n150["TargetRMSE"].astype(float)  # n150 in RMSE col
    out2 = SUBS / "tomorrow_2_v17pscal_n150.csv"
    cal_v17.to_csv(out2, index=False)
    log.info(f"   wrote {out2}")
    log.info(f"   TargetMBE mean: {cal_v17['TargetMBE'].mean():.3f}  "
             f"(v17 raw was {v17['TargetMBE'].mean():.3f})")

    # ── tomorrow_3: raw v17 / n150 split ────────────────────────────────────
    log.info("=" * 70)
    log.info("Bake 3: tomorrow_3_v17_n150_split.csv")
    log.info("        TargetMBE = v17 raw")
    log.info("        TargetRMSE = n150 raw")
    log.info("=" * 70)
    split = pd.DataFrame({
        "ID":         v17["ID"],
        "TargetMBE":  v17["TargetMBE"].astype(float),
        "TargetRMSE": n150["TargetRMSE"].astype(float),
    })
    out3 = SUBS / "tomorrow_3_v17_n150_split.csv"
    split.to_csv(out3, index=False)
    log.info(f"   wrote {out3}")

    # ── tomorrow_4: v17 standalone ──────────────────────────────────────────
    log.info("=" * 70)
    log.info("Bake 4: tomorrow_4_v17_standalone.csv")
    log.info("        TargetMBE = TargetRMSE = v17 raw")
    log.info("=" * 70)
    out4 = SUBS / "tomorrow_4_v17_standalone.csv"
    v17[["ID", "TargetMBE", "TargetRMSE"]].to_csv(out4, index=False)
    log.info(f"   wrote {out4}")

    log.info("=" * 70)
    log.info("Summary of baked submissions:")
    log.info(f"  1. {out1.name}  (highest expected payoff: calibrated n150)")
    log.info(f"  2. {out2.name}  (v17 with v17-calibration + n150 RMSE)")
    log.info(f"  3. {out3.name}  (raw split, no calibration)")
    log.info(f"  4. {out4.name}  (v17 standalone, both columns)")
    log.info("Upload in this order tomorrow.")


if __name__ == "__main__":
    main()
