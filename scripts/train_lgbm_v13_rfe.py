"""
v13: recursive-feature-elimination on top of v10.

Strategy (single-pass, not full iterative RFE — saves time):
  1. Train ONE LGBM on full training data with v10 features. No CV; we
     just need the gain-importance ranking.
  2. Sort features by gain. Drop the bottom THRESH% (default 30%).
     Keep all the "high gain" features.
  3. Run normal 6-fold CV + refit + save with the lean feature set.

Hypothesis: many of our 179 v10 features are redundant. The model should
be at least as accurate with the leaner set, and trains faster + uses
less memory (which helps when we later add pseudo-labels or LSA-SAF).

Run:
  bash /workspace/shell/run_train_v13_rfe.sh

Output:
  submissions/lgbm_v13_rfe.csv
  submissions/lgbm_v13_rfe_log.txt
  submissions/lgbm_v13_rfe_feature_ranking.csv  (full feature × gain table)
"""

import gc
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
SAMP  = ROOT / "data" / "raw" / "SampleSubmission.csv"

TARGET = "radiation (W/m2)"
ID     = "ID"
RUN_TAG = "lgbm_v13_rfe"

DROP_BOTTOM_FRAC = 0.30   # drop bottom 30% of features by gain


def score_components(y_true, y_pred):
    mbe  = float(np.mean(y_pred - y_true))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return abs(mbe), rmse, 0.5 * abs(mbe) + 0.5 * rmse


def add_time_features(df):
    ts = df["timestamp"]
    df["hour"]   = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["month"]  = ts.dt.month
    df["doy"]    = ts.dt.dayofyear
    df["year"]   = ts.dt.year
    df["dow"]    = ts.dt.dayofweek
    df["dom"]    = ts.dt.day
    df["hour_sin"] = np.sin(2 * np.pi * (df["hour"] + df["minute"]/60) / 24)
    df["hour_cos"] = np.cos(2 * np.pi * (df["hour"] + df["minute"]/60) / 24)
    df["doy_sin"]  = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"]  = np.cos(2 * np.pi * df["doy"] / 365.25)
    utc_hour = ts.dt.hour + ts.dt.minute / 60
    df["solar_clock"] = (utc_hour + df["longitude"] / 15.0) % 24
    df["hours_from_solar_noon"] = df["solar_clock"] - 12
    first = df.groupby("station")["timestamp"].transform("min")
    df["days_since_install"] = (df["timestamp"] - first).dt.total_seconds() / 86400
    return df


def ffill_cams(df):
    cams = [c for c in df.columns if c.startswith("ext_cams_")]
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)
    df[cams] = df.groupby("station")[cams].transform(lambda s: s.ffill().bfill())
    return df


def add_rolling_features(df):
    """Memory-efficient per-station rolling features (same as v8/v10)."""
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)
    roll_cols = {
        "temperature (degrees Celsius)": "temp",
        "relativehumidity (-)":           "rh",
        "precipitation (mm)":             "precip",
        "ext_era5_ssrd":                  "ssrd",
        "ext_era5_tcc":                   "tcc",
        "ext_om_ghi":                     "om_ghi",
        "ext_om_cc_total":                "om_cc",
        "ext_np_allsky_ghi":              "np_ghi",
    }
    windows = {"1h": 4, "3h": 12, "6h": 24, "24h": 96}
    g_idx = df.groupby("station", sort=False).indices

    def _assign(col_name, values):
        df[col_name] = values.astype(np.float32, copy=False)

    for col, short in roll_cols.items():
        if col not in df.columns:
            continue
        base = df[col].values.astype(np.float32, copy=False)
        for w_name, w_size in windows.items():
            out = np.empty_like(base, dtype=np.float32)
            for sta, idx in g_idx.items():
                s = pd.Series(base[idx])
                out[idx] = s.rolling(w_size, min_periods=1).mean().to_numpy(dtype=np.float32)
            _assign(f"{short}_mean_{w_name}", out)
            del out
        lag1 = np.empty_like(base, dtype=np.float32)
        for sta, idx in g_idx.items():
            lag1[idx] = pd.Series(base[idx]).shift(4).to_numpy(dtype=np.float32)
        _assign(f"{short}_lag_1h", lag1)
        lag3 = np.empty_like(base, dtype=np.float32)
        for sta, idx in g_idx.items():
            lag3[idx] = pd.Series(base[idx]).shift(12).to_numpy(dtype=np.float32)
        _assign(f"{short}_lag_3h", lag3)
        _assign(f"{short}_diff_1h", base - lag1)
        del base, lag1, lag3

    base_precip = df["precipitation (mm)"].values.astype(np.float32, copy=False)
    for label, w_size in [("precip_sum_6h", 24), ("precip_sum_24h", 96)]:
        out = np.empty_like(base_precip, dtype=np.float32)
        for sta, idx in g_idx.items():
            out[idx] = pd.Series(base_precip[idx]).rolling(w_size, min_periods=1).sum().to_numpy(dtype=np.float32)
        _assign(label, out)
        del out
    return df


def main():
    t0 = time.time()
    log.info("Loading data ...")
    train = pd.read_csv(TRAIN, parse_dates=["timestamp"])
    test  = pd.read_csv(TEST,  parse_dates=["timestamp"])

    for df in (train, test):
        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = df[c].astype(np.float32)

    train = add_time_features(ffill_cams(train))
    test  = add_time_features(ffill_cams(test))
    train = add_rolling_features(train)
    test  = add_rolling_features(test)

    for col in ["station", "country"]:
        cats = sorted(set(train[col].astype(str)) | set(test[col].astype(str)))
        train[col] = pd.Categorical(train[col].astype(str), categories=cats)
        test[col]  = pd.Categorical(test[col].astype(str), categories=cats)

    # Same feature set as v10
    BASE = [
        "precipitation (mm)", "relativehumidity (-)", "temperature (degrees Celsius)",
        "installation_height", "elevation", "latitude", "longitude",
        "ext_sol_elevation", "ext_sol_clearsky",
        "ext_lsa_dni", "ext_lsa_sid",
        "ext_era5_ssrd", "ext_era5_tcc", "ext_era5_tcwv", "ext_era5_blh", "ext_era5_sp",
        "ext_cams_aod550", "ext_cams_duaod550", "ext_cams_bcaod550",
        "ext_cams_omaod550", "ext_cams_suaod550", "ext_cams_ssaod550", "ext_cams_tcwv",
        "ext_pv_apparent_elevation", "ext_pv_etr", "ext_pv_linke_turbidity",
        "ext_pv_clearsky_ghi", "ext_pv_clearsky_dni", "ext_pv_clearsky_dhi",
        "ext_pv_clearsky_ghi_haur",
        "ext_om_ghi", "ext_om_direct_horiz", "ext_om_dni", "ext_om_dhi",
        "ext_om_cc_total", "ext_om_cc_low", "ext_om_cc_mid", "ext_om_cc_high",
        "ext_om_wind_speed_10m", "ext_om_wind_dir_10m",
        "ext_om_temperature_2m", "ext_om_dewpoint_2m", "ext_om_humidity_2m",
        "ext_om_pressure_surface", "ext_om_precip",
        "ext_np_allsky_ghi", "ext_np_allsky_dhi", "ext_np_allsky_dni",
        "ext_np_clrsky_ghi", "ext_np_cloud_amount", "ext_np_aod_550",
        "ext_np_precip_corr",
        "ext_fw_temp_lead_15m", "ext_fw_temp_lead_30m",
        "ext_fw_temp_lead_1h",  "ext_fw_temp_lead_3h",
        "ext_fw_temp_diff_1h",  "ext_fw_temp_diff_3h",
        "ext_fw_rh_lead_15m",   "ext_fw_rh_lead_30m",
        "ext_fw_rh_lead_1h",    "ext_fw_rh_lead_3h",
        "ext_fw_rh_diff_1h",
        "ext_fw_precip_lead_15m", "ext_fw_precip_lead_30m",
        "ext_fw_precip_lead_1h",  "ext_fw_precip_lead_3h",
        "ext_fw_precip_sum_lead_3h", "ext_fw_precip_sum_lead_24h",
        "ext_dd_temp_max", "ext_dd_temp_min", "ext_dd_temp_amp",
        "ext_dd_temp_mean", "ext_dd_temp_std",
        "ext_dd_rh_max", "ext_dd_rh_min", "ext_dd_rh_mean",
        "ext_dd_precip_sum", "ext_dd_precip_max",
        "ext_dd_om_ghi_max", "ext_dd_om_ghi_sum", "ext_dd_om_cc_mean",
        "ext_dd_np_ghi_max", "ext_dd_np_ghi_sum",
        "ext_dd_ssrd_max",   "ext_dd_ssrd_sum",
        "ext_eb_dT_x_blh_1h", "ext_eb_dT_x_blh_3h",
        "ext_eb_dRH_x_blh_1h",
        "ext_eb_T_minus_dewpoint", "ext_eb_water_content_proxy",
        "ext_eb_dT_per_W_per_m2",
        "ext_eb_morning_warming", "ext_eb_afternoon_cooling",
        "ext_csr_ghi", "ext_csr_bhi", "ext_csr_dhi", "ext_csr_bni",
        "ext_csr_clearsky_ghi", "ext_csr_clearsky_bhi",
        "ext_csr_clearsky_dhi", "ext_csr_clearsky_bni",
        "ext_csr_reliability",
        "ext_csr_clearness_kt", "ext_csr_diffuse_fraction",
    ]
    TIME = ["hour", "minute", "month", "doy", "year", "dow", "dom",
            "hour_sin", "hour_cos", "doy_sin", "doy_cos",
            "solar_clock", "hours_from_solar_noon", "days_since_install"]
    ROLLS = [c for c in train.columns
             if any(c.endswith(suf) for suf in
                    ["_mean_1h", "_mean_3h", "_mean_6h", "_mean_24h",
                     "_lag_1h", "_lag_3h", "_diff_1h", "_sum_6h", "_sum_24h"])
             and not c.startswith("ext_fw_") and not c.startswith("ext_eb_")]
    CATS = ["station", "country"]
    FULL_FEATURES = list(dict.fromkeys(BASE + TIME + ROLLS + CATS))
    log.info(f"Full feature set: {len(FULL_FEATURES)} features")

    X_train_full = train[FULL_FEATURES]
    y_train = train[TARGET].values
    X_test_full = test[FULL_FEATURES]

    params = dict(
        objective="regression", metric="rmse",
        learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
        verbose=-1, n_jobs=-1, seed=42,
        histogram_pool_size=512,
    )

    # ── Stage 1: quick one-shot training to get feature importance ────────────
    log.info("Stage 1: one-shot fit on full data to score feature importance ...")
    ds_full = lgb.Dataset(X_train_full, label=y_train, categorical_feature=CATS)
    quick = lgb.train(params, ds_full, num_boost_round=600)
    # LightGBM sanitizes feature names (replaces spaces/parens with underscores)
    # in feature_name(). Use the original FULL_FEATURES list to keep the
    # mapping to actual column names — they're in the same order as input.
    gain = pd.DataFrame({
        "feature": FULL_FEATURES,
        "gain":    quick.feature_importance(importance_type="gain"),
        "split":   quick.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    log.info(f"  top 10 by gain:")
    for _, r in gain.head(10).iterrows():
        log.info(f"    {r['feature']:40s}  gain={r['gain']:>14,.0f}")
    log.info(f"  bottom 10 by gain:")
    for _, r in gain.tail(10).iterrows():
        log.info(f"    {r['feature']:40s}  gain={r['gain']:>14,.0f}")

    rank_path = SUBS / f"{RUN_TAG}_feature_ranking.csv"
    gain.to_csv(rank_path, index=False)
    log.info(f"Saved ranking: {rank_path}")

    n_drop = int(len(gain) * DROP_BOTTOM_FRAC)
    KEEP = list(gain["feature"].iloc[:-n_drop])
    # Force-keep categoricals even if low gain
    for c in CATS:
        if c not in KEEP:
            KEEP.append(c)
    log.info(f"Dropping bottom {n_drop} of {len(gain)} → keeping {len(KEEP)} features")

    del ds_full, quick, X_train_full, X_test_full
    gc.collect()

    # ── Stage 2: full CV + refit with reduced feature set ─────────────────────
    X_train = train[KEEP]
    X_test  = test[KEEP]
    groups  = train["timestamp"].dt.month.values

    log.info("Stage 2: 6-fold CV with reduced feature set ...")
    fold_scores = []
    oof_pred = np.zeros(len(train))
    gkf = GroupKFold(n_splits=6)
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        ds_tr = lgb.Dataset(X_train.iloc[tr_idx], label=y_train[tr_idx], categorical_feature=CATS)
        ds_va = lgb.Dataset(X_train.iloc[va_idx], label=y_train[va_idx], categorical_feature=CATS)
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
                 f"best={model.best_iteration}")
        del ds_tr, ds_va, model, pred
        gc.collect()

    mbe_all, rmse_all, score_all = score_components(y_train, oof_pred)
    log.info(f"OOF: |MBE|={mbe_all:.2f}  RMSE={rmse_all:.2f}  score={score_all:.2f}")

    oof_df = pd.DataFrame({ID: train[ID].values,
                           "TargetMBE":  np.clip(oof_pred, 0, 1361),
                           "TargetRMSE": np.clip(oof_pred, 0, 1361)})
    oof_df.to_csv(SUBS / f"{RUN_TAG}_oof.csv", index=False)

    # ── Refit + predict ───────────────────────────────────────────────────────
    best_iter = int(np.mean([fs["best_iter"] for fs in fold_scores]))
    log.info(f"Refitting on full train for {best_iter} rounds ...")
    full_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=CATS)
    final = lgb.train(params, full_ds, num_boost_round=best_iter)

    log.info("Predicting test ...")
    test_pred = final.predict(X_test)

    # Per-(station, hour) night override
    train_night = pd.read_csv(TRAIN, usecols=["station", "timestamp", "ext_sol_elevation", TARGET],
                              parse_dates=["timestamp"])
    train_night = train_night[train_night["ext_sol_elevation"] <= 0].copy()
    train_night["hour"] = train_night["timestamp"].dt.hour
    per_hour_mean    = train_night.groupby(["station", "hour"])[TARGET].mean().to_dict()
    per_station_mean = train_night.groupby("station")[TARGET].mean().to_dict()
    fallback = float(np.mean(list(per_station_mean.values())) if per_station_mean else 0.0)

    is_night = test["ext_sol_elevation"] <= 0
    test_hours = test["timestamp"].dt.hour.values
    test_stations = test["station"].astype(str).values
    night_vals = np.array([per_hour_mean.get((s, h), per_station_mean.get(s, fallback))
                           for s, h in zip(test_stations, test_hours)])
    test_pred = np.where(is_night, night_vals, test_pred)
    test_pred = np.clip(test_pred, 0, 1361)

    sub = pd.DataFrame({ID: test[ID], "TargetMBE": test_pred, "TargetRMSE": test_pred})
    sample = pd.read_csv(SAMP)
    sub = sample[[ID]].merge(sub, on=ID, how="left")
    sub_path = SUBS / f"{RUN_TAG}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}  ({len(sub):,} rows)")

    log_path = SUBS / f"{RUN_TAG}_log.txt"
    log_path.write_text(json.dumps({
        "tag":            RUN_TAG,
        "full_features":  len(FULL_FEATURES),
        "kept_features":  len(KEEP),
        "drop_fraction":  DROP_BOTTOM_FRAC,
        "cv_folds":       fold_scores,
        "cv_oof_mbe":     round(mbe_all, 3),
        "cv_oof_rmse":    round(rmse_all, 3),
        "cv_oof_score":   round(score_all, 3),
        "best_iter":      best_iter,
        "submission":     str(sub_path),
        "wall_time_s":    round(time.time() - t0, 1),
        "dropped_features": list(gain["feature"].iloc[-n_drop:]),
    }, indent=2, default=str))
    log.info(f"Wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
