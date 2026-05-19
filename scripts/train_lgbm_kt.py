"""
TAHMO Solar Radiation — LGBM KT-Target (v1)
============================================
Predict the **clearness index** KT = radiation / clear-sky, then multiply
back by clear-sky to get W/m². The diurnal + latitudinal cycle is absorbed
into clear-sky (pure math); the model only learns the cloud/aerosol-driven
residual. Should reduce RMSE relative to predicting raw radiation directly.

Same nighttime override as the raw-radiation baseline (per-station mean).

Two paths at inference:
  ext_sol_elevation > 0 AND ext_sol_clearsky > KT_MIN_CLEARSKY  → pred = KT_hat × clearsky
  otherwise                                                    → pred = per-station night offset

Run:
  bash /workspace/shell/run_train_lgbm_kt.sh

Outputs:
  /workspace/submissions/lgbm_kt_v1.csv
  /workspace/submissions/lgbm_kt_v1_log.txt
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
RUN_TAG = "lgbm_kt_v1"

# Below this clear-sky value, KT is unstable (division near zero). These
# rows skip the daytime model entirely and use the nighttime override.
KT_MIN_CLEARSKY = 50.0
# Clip KT predictions to a plausible range — slightly above 1 allows
# cloud-edge enhancement (Section I showed 0.4% of training KT > 1).
KT_CLIP = (-0.05, 1.30)

# Same feature set as the raw baseline. clearsky stays because the model
# uses it to learn when KT departs from 1.
NUMERIC_FEATURES = [
    "precipitation (mm)", "relativehumidity (-)", "temperature (degrees Celsius)",
    "installation_height", "elevation", "latitude", "longitude",
    "ext_sol_elevation", "ext_sol_clearsky",
    "ext_lsa_dni", "ext_lsa_sid",
    "ext_era5_ssrd", "ext_era5_tcc", "ext_era5_tcwv", "ext_era5_blh", "ext_era5_sp",
    "ext_cams_aod550", "ext_cams_duaod550", "ext_cams_bcaod550",
]
CATEGORICAL_FEATURES = ["station", "country"]


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
    cams = ["ext_cams_aod550", "ext_cams_duaod550", "ext_cams_bcaod550"]
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)
    df[cams] = df.groupby("station")[cams].transform(lambda s: s.ffill().bfill())
    return df


def main():
    t0 = time.time()
    log.info("Loading data ...")
    train = pd.read_csv(TRAIN, parse_dates=["timestamp"])
    test  = pd.read_csv(TEST,  parse_dates=["timestamp"])
    log.info(f"  train: {train.shape}   test: {test.shape}")

    log.info("Adding time features and forward-filling CAMS ...")
    train = add_time_features(ffill_cadence_features(train))
    test  = add_time_features(ffill_cadence_features(test))

    for col in CATEGORICAL_FEATURES:
        cats = sorted(set(train[col].astype(str)) | set(test[col].astype(str)))
        train[col] = pd.Categorical(train[col].astype(str), categories=cats)
        test[col]  = pd.Categorical(test[col].astype(str),  categories=cats)

    TIME_FEATURES = ["hour", "minute", "month", "doy", "year",
                     "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    FEATURES = NUMERIC_FEATURES + TIME_FEATURES + CATEGORICAL_FEATURES

    # ── Restrict training to rows where KT is well-defined ──────────────────
    daytime_mask = train["ext_sol_clearsky"] > KT_MIN_CLEARSKY
    log.info(f"Training rows with clearsky > {KT_MIN_CLEARSKY}: "
             f"{daytime_mask.sum():,} / {len(train):,} "
             f"({daytime_mask.mean()*100:.1f}%)")

    train_day = train[daytime_mask].copy()
    train_day["kt"] = train_day[TARGET] / train_day["ext_sol_clearsky"]
    # Cap absurd KT values (sensor outliers > 2 etc.); training will be
    # more stable. KT_CLIP is the inference range, slightly wider here.
    train_day["kt"] = train_day["kt"].clip(-0.2, 2.0)

    X_train = train_day[FEATURES]
    y_train = train_day["kt"].values
    groups  = train_day["timestamp"].dt.month.values

    # ── CV: GroupKFold by month ──────────────────────────────────────────────
    log.info("Running 6-fold GroupKFold (by month) CV on KT target ...")
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
    oof_kt_pred = np.zeros(len(train_day))
    gkf = GroupKFold(n_splits=6)
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        ds_tr = lgb.Dataset(X_train.iloc[tr_idx], label=y_train[tr_idx],
                            categorical_feature=CATEGORICAL_FEATURES)
        ds_va = lgb.Dataset(X_train.iloc[va_idx], label=y_train[va_idx],
                            categorical_feature=CATEGORICAL_FEATURES)
        model = lgb.train(params, ds_tr, num_boost_round=2000, valid_sets=[ds_va],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        kt_pred = model.predict(X_train.iloc[va_idx])
        oof_kt_pred[va_idx] = kt_pred

        # Convert KT prediction to W/m² for scoring against actual radiation
        clearsky_va = train_day.iloc[va_idx]["ext_sol_clearsky"].values
        rad_pred = np.clip(kt_pred, *KT_CLIP) * clearsky_va
        rad_pred = np.clip(rad_pred, 0, 1361)
        rad_true = train_day.iloc[va_idx][TARGET].values

        held_out_month = groups[va_idx][0]
        mbe, rmse, score = score_components(rad_true, rad_pred)
        fold_scores.append({"fold": fold_idx, "month_held_out": int(held_out_month),
                            "mbe": mbe, "rmse": rmse, "score": score,
                            "best_iter": model.best_iteration})
        log.info(f"  fold {fold_idx} (month {held_out_month}):  "
                 f"|MBE|={mbe:6.2f}  RMSE={rmse:6.2f}  score={score:6.2f}  "
                 f"best_iter={model.best_iteration}")

    # OOF pooled (daytime only)
    clearsky_all = train_day["ext_sol_clearsky"].values
    oof_rad = np.clip(np.clip(oof_kt_pred, *KT_CLIP) * clearsky_all, 0, 1361)
    mbe_d, rmse_d, score_d = score_components(train_day[TARGET].values, oof_rad)
    log.info(f"OOF daytime only:  |MBE|={mbe_d:.2f}  RMSE={rmse_d:.2f}  score={score_d:.2f}")

    # ── Refit on all daytime training data ───────────────────────────────────
    best_iter = int(np.mean([fs["best_iter"] for fs in fold_scores]))
    log.info(f"Refitting on full daytime train for {best_iter} rounds ...")
    full_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES)
    final_model = lgb.train(params, full_ds, num_boost_round=best_iter)

    log.info("Predicting test KT ...")
    test_kt_pred = final_model.predict(test[FEATURES])
    test_kt_pred = np.clip(test_kt_pred, *KT_CLIP)
    test_rad_pred = test_kt_pred * test["ext_sol_clearsky"].values

    # ── Branch: night / twilight rows get per-station offset ─────────────────
    log.info("Applying per-station nighttime offset to night + twilight rows ...")
    if NIGHT.exists():
        night_table = pd.read_csv(NIGHT)
        night_map = dict(zip(night_table["station"].astype(str), night_table["mean_rad"]))
    else:
        log.warning("  night_offset_per_station.csv not found, using global 0")
        night_map = {}

    use_offset = (test["ext_sol_elevation"] <= 0) | (test["ext_sol_clearsky"] <= KT_MIN_CLEARSKY)
    offset_val = test["station"].astype(str).map(night_map).fillna(0.0).values
    test_rad_pred = np.where(use_offset, offset_val, test_rad_pred)
    log.info(f"  rows using offset (night + low-clearsky): {use_offset.sum():,} "
             f"({use_offset.mean()*100:.1f}%)")
    log.info(f"  rows using KT model:                     {(~use_offset).sum():,} "
             f"({(~use_offset).mean()*100:.1f}%)")

    # ── Final clip + write submission ────────────────────────────────────────
    test_rad_pred = np.clip(test_rad_pred, 0, 1361)

    sub = pd.DataFrame({
        ID: test[ID],
        "TargetMBE":  test_rad_pred,
        "TargetRMSE": test_rad_pred,
    })
    sample = pd.read_csv(SAMP)
    sub = sample[[ID]].merge(sub, on=ID, how="left")

    if sub[["TargetMBE", "TargetRMSE"]].isna().any().any():
        missing = sub[["TargetMBE", "TargetRMSE"]].isna().sum().sum()
        log.error(f"  {missing} predictions missing — check ID alignment")

    sub_path = SUBS / f"{RUN_TAG}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}  ({len(sub):,} rows)")

    log_path = SUBS / f"{RUN_TAG}_log.txt"
    info = {
        "tag":                   RUN_TAG,
        "target":                "KT (= radiation / clearsky) for daytime; per-station offset at night",
        "kt_min_clearsky":       KT_MIN_CLEARSKY,
        "kt_clip":               KT_CLIP,
        "wall_time_s":           round(time.time() - t0, 1),
        "n_train_daytime":       int(daytime_mask.sum()),
        "n_test":                int(len(test)),
        "features":              FEATURES,
        "lgbm_params":           params,
        "cv_folds":              fold_scores,
        "cv_oof_daytime_score":  round(score_d, 3),
        "cv_oof_daytime_mbe":    round(mbe_d, 3),
        "cv_oof_daytime_rmse":   round(rmse_d, 3),
        "final_n_rounds":        best_iter,
        "submission":            str(sub_path),
    }
    log_path.write_text(json.dumps(info, indent=2, default=str))
    log.info(f"Saved run log:   {log_path}")
    log.info(f"Wall time: {info['wall_time_s']}s")


if __name__ == "__main__":
    main()
