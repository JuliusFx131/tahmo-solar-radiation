"""
TAHMO Solar Radiation — LGBM Baseline (v1)
==========================================
Single shared LightGBM regressor on raw 15-min radiation. The plan that drives
this script came from the visualization notebook (Sections J, K, M, N):

  • Forum-confirmed: training data is RAW; cleaning hurts LB → predict raw.
  • Per-station nighttime offset (from Section J) overrides the model's
    nighttime prediction. Aligns with "mimic the sensor" strategy.
  • TA00338 is defective — handled by the shared model + nighttime override;
    no special-case code needed beyond that.
  • CAMS is 3-hourly → forward-fill within station before fitting.
  • CV by month (odd training months → 6 folds) mimics the test structure.
  • Composite score:   0.5·|MBE| + 0.5·RMSE.

Run:
  bash /workspace/shell/run_train_lgbm.sh

Outputs:
  /workspace/submissions/lgbm_baseline_v1.csv          # Zindi submission format
  /workspace/submissions/lgbm_baseline_v1_log.txt      # CV scores + run info
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT  = Path(__file__).resolve().parent.parent
PROC  = ROOT / "data" / "processed"
SUBS  = ROOT / "submissions"
SUBS.mkdir(parents=True, exist_ok=True)

TRAIN = PROC / "Train_enhanced.csv"
TEST  = PROC / "Test_enhanced.csv"
NIGHT = PROC / "night_offset_per_station.csv"
SAMP  = ROOT / "data" / "raw" / "SampleSubmission.csv"

TARGET = "radiation (W/m2)"
ID     = "ID"
RUN_TAG = "lgbm_baseline_v1"

# ── Feature set ──────────────────────────────────────────────────────────────
# Kept based on Section N correlations + redundancy analysis:
#   - Drop ext_sol_zenith (collinear with ext_sol_elevation)
#   - Drop ext_sol_eqtime, azimuth, hour_angle, declination, earth_sun_dist,
#     daylight, day_length (each |r|<0.10 with target)
#   - Drop ext_lsa_sis (collinear with ext_era5_ssrd)
NUMERIC_FEATURES = [
    "precipitation (mm)", "relativehumidity (-)", "temperature (degrees Celsius)",
    "installation_height", "elevation", "latitude", "longitude",
    "ext_sol_elevation", "ext_sol_clearsky",
    "ext_lsa_dni", "ext_lsa_sid",
    "ext_era5_ssrd", "ext_era5_tcc", "ext_era5_tcwv", "ext_era5_blh", "ext_era5_sp",
    "ext_cams_aod550", "ext_cams_duaod550", "ext_cams_bcaod550",
]
CATEGORICAL_FEATURES = ["station", "country"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def score_components(y_true, y_pred):
    mbe  = float(np.mean(y_pred - y_true))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return abs(mbe), rmse, 0.5 * abs(mbe) + 0.5 * rmse


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df["hour"]   = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["month"]  = ts.dt.month
    df["doy"]    = ts.dt.dayofyear
    df["year"]   = ts.dt.year
    df["hour_sin"] = np.sin(2 * np.pi * (df["hour"] + df["minute"]/60) / 24)
    df["hour_cos"] = np.cos(2 * np.pi * (df["hour"] + df["minute"]/60) / 24)
    df["doy_sin"]  = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"]  = np.cos(2 * np.pi * df["doy"] / 365.25)
    return df


def ffill_cadence_features(df: pd.DataFrame) -> pd.DataFrame:
    """CAMS is 3-hourly → forward+back-fill within station so the model sees
    a value for every 15-min row instead of NaN 67% of the time."""
    cams = ["ext_cams_aod550", "ext_cams_duaod550", "ext_cams_bcaod550"]
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)
    df[cams] = df.groupby("station")[cams].transform(lambda s: s.ffill().bfill())
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("Loading data ...")
    train = pd.read_csv(TRAIN, parse_dates=["timestamp"])
    test  = pd.read_csv(TEST,  parse_dates=["timestamp"])
    log.info(f"  train: {train.shape}   test: {test.shape}")

    log.info("Adding time features and forward-filling CAMS ...")
    train = add_time_features(ffill_cadence_features(train))
    test  = add_time_features(ffill_cadence_features(test))

    # Cast categoricals
    for col in CATEGORICAL_FEATURES:
        cats = sorted(set(train[col].astype(str)) | set(test[col].astype(str)))
        train[col] = pd.Categorical(train[col].astype(str), categories=cats)
        test[col]  = pd.Categorical(test[col].astype(str),  categories=cats)

    TIME_FEATURES = ["hour", "minute", "month", "doy", "year",
                     "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    FEATURES = NUMERIC_FEATURES + TIME_FEATURES + CATEGORICAL_FEATURES

    X_train = train[FEATURES]
    y_train = train[TARGET].values
    X_test  = test[FEATURES]
    groups  = train["timestamp"].dt.month.values  # GroupKFold by month

    # ── CV: GroupKFold by month ──────────────────────────────────────────────
    log.info("Running 6-fold GroupKFold (by month) CV ...")
    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=100,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        verbose=-1,
        n_jobs=-1,
        seed=42,
    )

    fold_scores = []
    oof_pred = np.zeros(len(train))
    gkf = GroupKFold(n_splits=6)
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        ds_tr = lgb.Dataset(X_train.iloc[tr_idx], label=y_train[tr_idx],
                            categorical_feature=CATEGORICAL_FEATURES)
        ds_va = lgb.Dataset(X_train.iloc[va_idx], label=y_train[va_idx],
                            categorical_feature=CATEGORICAL_FEATURES)
        model = lgb.train(params, ds_tr, num_boost_round=2000, valid_sets=[ds_va],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        pred = model.predict(X_train.iloc[va_idx])
        oof_pred[va_idx] = pred
        held_out_month = groups[va_idx][0]
        mbe, rmse, score = score_components(y_train[va_idx], pred)
        fold_scores.append({"fold": fold_idx, "month_held_out": int(held_out_month),
                            "mbe": mbe, "rmse": rmse, "score": score,
                            "best_iter": model.best_iteration})
        log.info(f"  fold {fold_idx} (month {held_out_month}):  "
                 f"|MBE|={mbe:6.2f}  RMSE={rmse:6.2f}  score={score:6.2f}  "
                 f"best_iter={model.best_iteration}")

    mbe_all, rmse_all, score_all = score_components(y_train, oof_pred)
    log.info(f"OOF (all folds pooled):  |MBE|={mbe_all:.2f}  RMSE={rmse_all:.2f}  "
             f"score={score_all:.2f}")

    # ── Refit on full training data ──────────────────────────────────────────
    best_iter = int(np.mean([fs["best_iter"] for fs in fold_scores]))
    log.info(f"Refitting on full train for {best_iter} rounds ...")
    full_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES)
    final_model = lgb.train(params, full_ds, num_boost_round=best_iter)

    log.info("Predicting test ...")
    test_pred = final_model.predict(X_test)

    # ── Nighttime override: per-station mean from training (Section J) ───────
    log.info("Applying per-station nighttime offset ...")
    if NIGHT.exists():
        night_table = pd.read_csv(NIGHT)
        night_map = dict(zip(night_table["station"].astype(str), night_table["mean_rad"]))
    else:
        log.warning("  night_offset_per_station.csv not found, using global 0")
        night_map = {}

    is_night = test["ext_sol_elevation"] <= 0
    night_override = test["station"].astype(str).map(night_map).fillna(0.0)
    test_pred = np.where(is_night, night_override.values, test_pred)
    log.info(f"  rows overridden as night: {is_night.sum():,} ({is_night.mean()*100:.1f}%)")

    # ── Clip + format ────────────────────────────────────────────────────────
    test_pred = np.clip(test_pred, 0, 1361)

    sub = pd.DataFrame({
        ID: test[ID],
        "TargetMBE":  test_pred,
        "TargetRMSE": test_pred,
    })

    # Reorder to match SampleSubmission row order
    sample = pd.read_csv(SAMP)
    sub = sample[[ID]].merge(sub, on=ID, how="left")

    if sub[["TargetMBE", "TargetRMSE"]].isna().any().any():
        missing = sub[["TargetMBE", "TargetRMSE"]].isna().sum().sum()
        log.error(f"  {missing} predictions missing — check ID alignment")

    sub_path = SUBS / f"{RUN_TAG}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}  ({len(sub):,} rows)")

    # ── Log file ─────────────────────────────────────────────────────────────
    log_path = SUBS / f"{RUN_TAG}_log.txt"
    info = {
        "tag":            RUN_TAG,
        "wall_time_s":    round(time.time() - t0, 1),
        "n_train":        int(len(train)),
        "n_test":         int(len(test)),
        "features":       FEATURES,
        "lgbm_params":    params,
        "cv_folds":       fold_scores,
        "cv_oof_score":   round(score_all, 3),
        "cv_oof_mbe":     round(mbe_all, 3),
        "cv_oof_rmse":    round(rmse_all, 3),
        "final_n_rounds": best_iter,
        "submission":     str(sub_path),
    }
    log_path.write_text(json.dumps(info, indent=2, default=str))
    log.info(f"Saved run log:   {log_path}")
    log.info(f"Wall time: {info['wall_time_s']}s")


if __name__ == "__main__":
    main()
