"""
TAHMO Solar Radiation — LGBM + station-hour climatology (v1)
=============================================================
Same features as train_lgbm_feat_eng.py PLUS a per-(station, hour-of-day)
climatology feature: the mean training-set radiation for that station at
that UTC hour. Strong target encoding that gives the model a direct
"typical radiation here at this time" anchor.

LEAKAGE NOTE: the climatology is computed *only on the in-fold training
rows* during CV, then re-computed on the full training set for the final
model. Test rows get the full-train climatology.

Same shared LGBM architecture as the raw baseline, with feature engineering
focused on the things our prior runs didn't have:

  • Local solar time (hours from solar noon at the station's longitude)
  • Days since installation (per station) — captures any early drift
  • Day-of-week, day-of-month
  • Weather lags + rolling stats: temp, humidity, precip
    (past 1h / 3h / 24h, computed per-station respecting time order)
  • ERA5 ssrd lag (past 1h, 3h) — cloud-front detection
  • Recent precip accumulation (past 6h, 24h)
  • Temperature & humidity change vs 1h ago

Rolling features are computed on the full (train + test) timeline per
station, so test rows see real lags from their adjacent training-month
rows. No radiation lag (target is hidden in test).

Per-station nighttime override stays the same (Section J table).
No per-station calibration (the v1 calib slightly hurt on LB).

Run:
  bash /workspace/shell/run_train_lgbm_climatology.sh

Outputs:
  /workspace/submissions/lgbm_climatology_v1.csv
  /workspace/submissions/lgbm_climatology_v1_log.txt
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
RUN_TAG = "lgbm_climatology_v1"

CATEGORICAL_FEATURES = ["station", "country"]


def score_components(y_true, y_pred):
    mbe  = float(np.mean(y_pred - y_true))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return abs(mbe), rmse, 0.5 * abs(mbe) + 0.5 * rmse


def add_basic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df["hour"]   = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["month"]  = ts.dt.month
    df["doy"]    = ts.dt.dayofyear
    df["year"]   = ts.dt.year
    df["dow"]    = ts.dt.dayofweek      # NEW
    df["dom"]    = ts.dt.day            # NEW
    df["hour_sin"] = np.sin(2 * np.pi * (df["hour"] + df["minute"]/60) / 24)
    df["hour_cos"] = np.cos(2 * np.pi * (df["hour"] + df["minute"]/60) / 24)
    df["doy_sin"]  = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"]  = np.cos(2 * np.pi * df["doy"] / 365.25)
    return df


def add_solar_clock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hours from local solar noon: corrects UTC for longitude so stations
    across the African continent share a comparable midday reference."""
    utc_hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    local_hour = (utc_hour + df["longitude"] / 15.0) % 24
    df["solar_clock"] = local_hour
    df["hours_from_solar_noon"] = local_hour - 12.0
    return df


def add_days_since_install(df: pd.DataFrame) -> pd.DataFrame:
    """For each station, days from its first observation in the combined
    train+test timeline. Captures sensor age."""
    first_seen = df.groupby("station")["timestamp"].transform("min")
    df["days_since_install"] = (df["timestamp"] - first_seen).dt.total_seconds() / 86400
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-station rolling means / lags for weather + ERA5 features.
    Computed on the full timeline so test rows pick up valid lags from
    their adjacent training-month rows."""
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)

    # Columns to roll — only those known at inference time (no radiation lag)
    roll_cols = {
        "temperature (degrees Celsius)": "temp",
        "relativehumidity (-)":           "rh",
        "precipitation (mm)":             "precip",
        "ext_era5_ssrd":                  "ssrd",
        "ext_era5_tcc":                   "tcc",
    }

    # 15-min cadence: 4 rows = 1h, 12 = 3h, 24 = 6h, 96 = 24h
    windows = {"1h": 4, "3h": 12, "6h": 24, "24h": 96}
    grouped = df.groupby("station", group_keys=False)

    new_blocks = {}
    for col, short in roll_cols.items():
        for w_name, w_size in windows.items():
            roll = grouped[col].rolling(w_size, min_periods=1).mean().reset_index(level=0, drop=True)
            new_blocks[f"{short}_mean_{w_name}"] = roll
        # Lag 1h and 3h
        new_blocks[f"{short}_lag_1h"] = grouped[col].shift(4).reset_index(level=0, drop=True)
        new_blocks[f"{short}_lag_3h"] = grouped[col].shift(12).reset_index(level=0, drop=True)
        # Diff vs 1h ago (change-rate signal)
        new_blocks[f"{short}_diff_1h"] = df[col] - new_blocks[f"{short}_lag_1h"]

    # Cumulative precip past 6h and 24h (sum, not mean)
    new_blocks["precip_sum_6h"]  = grouped["precipitation (mm)"].rolling(24, min_periods=1).sum().reset_index(level=0, drop=True)
    new_blocks["precip_sum_24h"] = grouped["precipitation (mm)"].rolling(96, min_periods=1).sum().reset_index(level=0, drop=True)

    rolled = pd.DataFrame(new_blocks)
    return pd.concat([df, rolled], axis=1)


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

    # Mark splits, concatenate, build features on the combined timeline.
    train["_split"] = "train"
    test["_split"]  = "test"
    test[TARGET]    = np.nan
    full = pd.concat([train, test], ignore_index=True)

    log.info("Building features (basic time, solar clock, lags, rolls, ffill) ...")
    full = ffill_cadence_features(full)
    full = add_basic_time_features(full)
    full = add_solar_clock_features(full)
    full = add_days_since_install(full)
    full = add_rolling_features(full)
    log.info(f"  full shape after FE: {full.shape}")

    # Re-split
    train = full[full["_split"] == "train"].copy()
    test  = full[full["_split"] == "test"].copy()
    train = train.sort_values(["station", "timestamp"]).reset_index(drop=True)
    test  = test.sort_values(["station", "timestamp"]).reset_index(drop=True)

    for col in CATEGORICAL_FEATURES:
        cats = sorted(set(train[col].astype(str)) | set(test[col].astype(str)))
        train[col] = pd.Categorical(train[col].astype(str), categories=cats)
        test[col]  = pd.Categorical(test[col].astype(str),  categories=cats)

    # Build the feature list (everything we engineered + the original ext_* + raw weather)
    BASE = [
        "precipitation (mm)", "relativehumidity (-)", "temperature (degrees Celsius)",
        "installation_height", "elevation", "latitude", "longitude",
        "ext_sol_elevation", "ext_sol_clearsky",
        "ext_lsa_dni", "ext_lsa_sid",
        "ext_era5_ssrd", "ext_era5_tcc", "ext_era5_tcwv", "ext_era5_blh", "ext_era5_sp",
        "ext_cams_aod550", "ext_cams_duaod550", "ext_cams_bcaod550",
    ]
    TIME = ["hour", "minute", "month", "doy", "year", "dow", "dom",
            "hour_sin", "hour_cos", "doy_sin", "doy_cos",
            "solar_clock", "hours_from_solar_noon", "days_since_install"]
    ROLLS = [c for c in train.columns if any(c.endswith(suf)
             for suf in ["_mean_1h", "_mean_3h", "_mean_6h", "_mean_24h",
                         "_lag_1h", "_lag_3h", "_diff_1h", "_sum_6h", "_sum_24h"])]
    CLIM = ["station_hour_clim"]
    FEATURES = BASE + TIME + ROLLS + CLIM + CATEGORICAL_FEATURES
    log.info(f"  features used: {len(FEATURES)}  ({len(ROLLS)} rolling/lag, 1 climatology)")

    # Climatology helper. Compute on a training subset, apply to any df.
    def compute_clim(src_df, fallback_mean):
        """Return Series mapping (station, hour) → mean of TARGET."""
        sub = src_df[[TARGET]].copy()
        sub["station"] = src_df["station"].astype(str)
        sub["hour"]    = src_df["hour"].values
        return sub.groupby(["station", "hour"])[TARGET].mean(), fallback_mean

    def apply_clim(target_df, clim_series, fallback):
        keys = list(zip(target_df["station"].astype(str), target_df["hour"]))
        return np.array([clim_series.get(k, fallback) for k in keys], dtype=np.float32)

    # We'll fill `station_hour_clim` per-fold inside the CV loop and once
    # globally for the final model. Pre-allocate the column.
    train["station_hour_clim"] = np.nan
    test["station_hour_clim"]  = np.nan

    y_train = train[TARGET].values
    groups  = train["timestamp"].dt.month.values

    params = dict(
        objective="regression", metric="rmse",
        learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
        verbose=-1, n_jobs=-1, seed=42,
    )

    log.info("Running 6-fold GroupKFold (by month) CV "
             "(climatology refit per fold) ...")
    fold_scores = []
    oof_pred = np.zeros(len(train))
    gkf = GroupKFold(n_splits=6)
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(train, y_train, groups)):
        # Build climatology from this fold's training rows only (no leakage).
        tr_view = train.iloc[tr_idx]
        clim_series, _ = compute_clim(tr_view, fallback_mean=tr_view[TARGET].mean())
        fallback = float(tr_view[TARGET].mean())

        # Inject the feature in place — each fold overwrites the previous one
        # so there's no cross-fold leakage, and LGBM copies the data internally
        # at lgb.train time so modifying after is safe.
        train.loc[train.index[tr_idx], "station_hour_clim"] = apply_clim(train.iloc[tr_idx], clim_series, fallback)
        train.loc[train.index[va_idx], "station_hour_clim"] = apply_clim(train.iloc[va_idx], clim_series, fallback)
        X_train = train[FEATURES]

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
    log.info(f"OOF:  |MBE|={mbe_all:.2f}  RMSE={rmse_all:.2f}  score={score_all:.2f}")

    best_iter = int(np.mean([fs["best_iter"] for fs in fold_scores]))
    log.info(f"Refitting on full train for {best_iter} rounds ...")
    # Final climatology: from ALL training rows.
    clim_full, _ = compute_clim(train, fallback_mean=train[TARGET].mean())
    fb_full = float(train[TARGET].mean())
    train["station_hour_clim"] = apply_clim(train, clim_full, fb_full)
    test["station_hour_clim"]  = apply_clim(test,  clim_full, fb_full)
    X_train_full = train[FEATURES]
    X_test       = test[FEATURES]

    full_ds = lgb.Dataset(X_train_full, label=y_train, categorical_feature=CATEGORICAL_FEATURES)
    final_model = lgb.train(params, full_ds, num_boost_round=best_iter)

    log.info("Predicting test ...")
    test_pred = final_model.predict(X_test)

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
    test_pred = np.clip(test_pred, 0, 1361)

    sub = pd.DataFrame({ID: test[ID], "TargetMBE": test_pred, "TargetRMSE": test_pred})
    sample = pd.read_csv(SAMP)
    sub = sample[[ID]].merge(sub, on=ID, how="left")
    if sub[["TargetMBE", "TargetRMSE"]].isna().any().any():
        log.error("  predictions missing — check ID alignment")

    sub_path = SUBS / f"{RUN_TAG}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}  ({len(sub):,} rows)")

    # Feature importance for inspection
    imp = pd.DataFrame({
        "feature":   final_model.feature_name(),
        "gain":      final_model.feature_importance(importance_type="gain"),
        "split":     final_model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    log.info("Top 15 features by gain:")
    for _, row in imp.head(15).iterrows():
        log.info(f"  {row['feature']:40s}  gain={row['gain']:>12,.0f}  split={row['split']:>5}")

    log_path = SUBS / f"{RUN_TAG}_log.txt"
    info = {
        "tag":              RUN_TAG,
        "wall_time_s":      round(time.time() - t0, 1),
        "n_train":          int(len(train)),
        "n_test":           int(len(test)),
        "n_features":       len(FEATURES),
        "features":         FEATURES,
        "lgbm_params":      params,
        "cv_folds":         fold_scores,
        "cv_oof_mbe":       round(mbe_all, 3),
        "cv_oof_rmse":      round(rmse_all, 3),
        "cv_oof_score":     round(score_all, 3),
        "final_n_rounds":   best_iter,
        "submission":       str(sub_path),
        "feature_importance_top15": imp.head(15).to_dict("records"),
    }
    log_path.write_text(json.dumps(info, indent=2, default=str))
    log.info(f"Saved run log:   {log_path}")
    log.info(f"Wall time: {info['wall_time_s']}s")


if __name__ == "__main__":
    main()
