"""
TAHMO Solar Radiation — LGBM + extended external data (v3)
===========================================================
Builds on feat_eng_v1 by **using the new external sources we just added**:
pvlib clear-sky variants and Open-Meteo's GHI/DNI/DHI + layered cloud cover
+ wind + dewpoint (and dropping `ext_om_cape` which came back 0% covered).

What's added vs v1:
  • pvlib features: apparent_elevation, etr, linke_turbidity, clearsky GHI/DNI/DHI
    (both Ineichen-Perez and Haurwitz variants). Skipped airmass_* (50% NaN at
    night) and apparent_zenith (collinear with apparent_elevation).
  • Open-Meteo: GHI / DNI / DHI / direct-on-horizontal as direct radiation
    estimates, layered cloud cover (low/mid/high) which the LB-leader called
    out as useful, wind speed/direction, dewpoint, humidity, surface pressure.
  • Rolling features for `ext_om_ghi` and `ext_om_cc_total` (1h / 3h / 6h / 24h)
    — Open-Meteo's GHI is a fresh radiation estimate independent of ERA5.

What stays from v1:
  • Local solar clock, days-since-install, dow/dom
  • Per-(station, hour) night override (better than per-station mean)
  • Single LGBM with station + country as categoricals
  • 6-fold GroupKFold-by-month CV

Run:
  bash /workspace/shell/run_train_lgbm_feat_eng_v3.sh

Outputs:
  /workspace/submissions/lgbm_feat_eng_v3.csv
  /workspace/submissions/lgbm_feat_eng_v3_log.txt
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
RUN_TAG = "lgbm_feat_eng_v3"

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

    # Columns to roll — only those known at inference time (no radiation lag).
    # v3 adds Open-Meteo GHI and total cloud cover (independent of ERA5).
    roll_cols = {
        "temperature (degrees Celsius)": "temp",
        "relativehumidity (-)":           "rh",
        "precipitation (mm)":             "precip",
        "ext_era5_ssrd":                  "ssrd",
        "ext_era5_tcc":                   "tcc",
        "ext_om_ghi":                     "om_ghi",
        "ext_om_cc_total":                "om_cc",
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

    # Downcast float64 → float32 to halve memory before we concat + roll.
    # Container is cgroup-capped at 8GB; v3 has 60 input cols + ~50 rolling cols.
    log.info("Downcasting float64 → float32 ...")
    for df in (train, test):
        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = df[c].astype("float32")

    # Mark splits, concatenate, build features on the combined timeline.
    train["_split"] = "train"
    test["_split"]  = "test"
    test[TARGET]    = np.float32(np.nan)
    full = pd.concat([train, test], ignore_index=True, copy=False)

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
        # v3: pvlib (drop airmass_* — 50% NaN at night; drop apparent_zenith — collinear with elevation)
        "ext_pv_apparent_elevation", "ext_pv_etr", "ext_pv_linke_turbidity",
        "ext_pv_clearsky_ghi", "ext_pv_clearsky_dni", "ext_pv_clearsky_dhi",
        "ext_pv_clearsky_ghi_haur",
        # v3: Open-Meteo (drop cape — 0% coverage)
        "ext_om_ghi", "ext_om_direct_horiz", "ext_om_dni", "ext_om_dhi",
        "ext_om_cc_total", "ext_om_cc_low", "ext_om_cc_mid", "ext_om_cc_high",
        "ext_om_wind_speed_10m", "ext_om_wind_dir_10m",
        "ext_om_temperature_2m", "ext_om_dewpoint_2m", "ext_om_humidity_2m",
        "ext_om_pressure_surface", "ext_om_precip",
    ]
    TIME = ["hour", "minute", "month", "doy", "year", "dow", "dom",
            "hour_sin", "hour_cos", "doy_sin", "doy_cos",
            "solar_clock", "hours_from_solar_noon", "days_since_install"]
    ROLLS = [c for c in train.columns if any(c.endswith(suf)
             for suf in ["_mean_1h", "_mean_3h", "_mean_6h", "_mean_24h",
                         "_lag_1h", "_lag_3h", "_diff_1h", "_sum_6h", "_sum_24h"])]
    FEATURES = BASE + TIME + ROLLS + CATEGORICAL_FEATURES
    log.info(f"  features used: {len(FEATURES)}  ({len(ROLLS)} rolling/lag)")

    X_train = train[FEATURES]
    y_train = train[TARGET].values
    X_test  = test[FEATURES]
    groups  = train["timestamp"].dt.month.values

    params = dict(
        objective="regression", metric="rmse",
        learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
        verbose=-1, n_jobs=-1, seed=42,
    )

    log.info("Running 6-fold GroupKFold (by month) CV ...")
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
    log.info(f"OOF:  |MBE|={mbe_all:.2f}  RMSE={rmse_all:.2f}  score={score_all:.2f}")

    best_iter = int(np.mean([fs["best_iter"] for fs in fold_scores]))
    log.info(f"Refitting on full train for {best_iter} rounds ...")
    full_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES)
    final_model = lgb.train(params, full_ds, num_boost_round=best_iter)

    log.info("Predicting test ...")
    test_pred = final_model.predict(X_test)

    # Per-(station, hour) nighttime override — built only from training rows
    # with elev<=0, so seasonal daylight shifts come along for free.
    log.info("Building per-(station, hour) night override from training rows ...")
    train_night_src = pd.read_csv(TRAIN,
                                  usecols=["station", "timestamp", "ext_sol_elevation", TARGET],
                                  parse_dates=["timestamp"])
    train_night_src = train_night_src[train_night_src["ext_sol_elevation"] <= 0].copy()
    train_night_src["hour"] = train_night_src["timestamp"].dt.hour
    per_hour_mean    = train_night_src.groupby(["station", "hour"])[TARGET].mean().to_dict()
    per_station_mean = train_night_src.groupby("station")[TARGET].mean().to_dict()
    fallback = float(np.mean(list(per_station_mean.values())) if per_station_mean else 0.0)

    is_night = test["ext_sol_elevation"] <= 0
    test_hours = test["timestamp"].dt.hour.values
    test_stations = test["station"].astype(str).values
    night_vals = np.array([
        per_hour_mean.get((s, h), per_station_mean.get(s, fallback))
        for s, h in zip(test_stations, test_hours)
    ])
    test_pred = np.where(is_night, night_vals, test_pred)
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
