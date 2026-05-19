"""
TAHMO Solar Radiation — v17 clearsky-weighted training
========================================================
Single LGBM on v10 features, but trained with sample weights proportional to
ext_csr_clearsky_ghi. The motivation: the LB metric (RMSE) is dominated by
absolute errors at high radiation. By up-weighting high-clearsky rows during
training, the optimization focuses where the leaderboard penalty is biggest.

This is what predicting KT (kt_v1) was SUPPOSED to achieve but did the
opposite of: with target = y/x, RMSE-in-y = sqrt(mean((kt_err)² × x²)) — the
KT model implicitly DOWN-weights the high-clearsky rows. Here we up-weight
them directly, on the raw target, so the loss surface aligns with the LB.

  weights = clip(ext_csr_clearsky_ghi / 1000, 0.05, 1.0)
            └────────────────────┬────────────────────┘
                  ~1.0 at noon clear-sky
                  ~0.05 at night / deep dawn (floor so night rows
                  still contribute a bit to the model)

Everything else (CV, refit, night override, submission format) matches v10.

Run:
  bash /workspace/shell/run_train_v17_wsky.sh

Outputs:
  /workspace/submissions/lgbm_v17_wsky.csv
  /workspace/submissions/lgbm_v17_wsky_log.txt
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
RUN_TAG = "lgbm_v17_wsky"

CATEGORICAL_FEATURES = ["station", "country"]

# Sample-weighting parameters
WEIGHT_SCALE = 1000.0   # divide clearsky_ghi by this to get nominal [0,1]
WEIGHT_FLOOR = 0.05     # minimum weight so night rows still contribute
WEIGHT_COL   = "ext_csr_clearsky_ghi"  # strongest clearsky signal we have


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
    df["dow"]    = ts.dt.dayofweek
    df["dom"]    = ts.dt.day
    df["hour_sin"] = np.sin(2 * np.pi * (df["hour"] + df["minute"]/60) / 24)
    df["hour_cos"] = np.cos(2 * np.pi * (df["hour"] + df["minute"]/60) / 24)
    df["doy_sin"]  = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"]  = np.cos(2 * np.pi * df["doy"] / 365.25)
    return df


def add_solar_clock_features(df: pd.DataFrame) -> pd.DataFrame:
    utc_hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    local_hour = (utc_hour + df["longitude"] / 15.0) % 24
    df["solar_clock"] = local_hour
    df["hours_from_solar_noon"] = local_hour - 12.0
    return df


def add_days_since_install(df: pd.DataFrame) -> pd.DataFrame:
    first_seen = df.groupby("station")["timestamp"].transform("min")
    df["days_since_install"] = (df["timestamp"] - first_seen).dt.total_seconds() / 86400
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
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

    def _assign(col_name: str, values: np.ndarray):
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


def ffill_cadence_features(df: pd.DataFrame) -> pd.DataFrame:
    cams = [
        "ext_cams_aod550", "ext_cams_duaod550", "ext_cams_bcaod550",
        "ext_cams_omaod550", "ext_cams_suaod550", "ext_cams_ssaod550",
        "ext_cams_tcwv",
    ]
    cams = [c for c in cams if c in df.columns]
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)
    df[cams] = df.groupby("station")[cams].transform(lambda s: s.ffill().bfill())
    return df


def main():
    t0 = time.time()
    log.info("Loading data ...")
    train = pd.read_csv(TRAIN, parse_dates=["timestamp"])
    test  = pd.read_csv(TEST,  parse_dates=["timestamp"])
    log.info(f"  train: {train.shape}   test: {test.shape}")

    log.info("Downcasting float64 → float32 ...")
    for df in (train, test):
        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = df[c].astype("float32")

    train["_split"] = "train"
    test["_split"]  = "test"
    test[TARGET]    = np.float32(np.nan)
    full = pd.concat([train, test], ignore_index=True, copy=False)

    log.info("Building features ...")
    full = ffill_cadence_features(full)
    full = add_basic_time_features(full)
    full = add_solar_clock_features(full)
    full = add_days_since_install(full)
    full = add_rolling_features(full)
    log.info(f"  full shape after FE: {full.shape}")

    train = full[full["_split"] == "train"].copy()
    test  = full[full["_split"] == "test"].copy()
    train = train.sort_values(["station", "timestamp"]).reset_index(drop=True)
    test  = test.sort_values(["station", "timestamp"]).reset_index(drop=True)
    del full
    gc.collect()

    for col in CATEGORICAL_FEATURES:
        cats = sorted(set(train[col].astype(str)) | set(test[col].astype(str)))
        train[col] = pd.Categorical(train[col].astype(str), categories=cats)
        test[col]  = pd.Categorical(test[col].astype(str),  categories=cats)

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
        "ext_dd_om_ghi_max", "ext_dd_om_ghi_sum",
        "ext_dd_om_cc_mean",
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
             and not c.startswith("ext_fw_")]
    BASE_PRESENT = [c for c in BASE if c in train.columns]
    if len(BASE_PRESENT) != len(BASE):
        missing = set(BASE) - set(BASE_PRESENT)
        log.warning(f"  missing BASE features (skipped): {sorted(missing)}")
    FEATURES = list(dict.fromkeys(BASE_PRESENT + TIME + ROLLS + CATEGORICAL_FEATURES))
    log.info(f"  features used: {len(FEATURES)}  ({len(ROLLS)} rolling/lag)")

    X_train = train[FEATURES]
    y_train = train[TARGET].values
    X_test  = test[FEATURES]
    groups  = train["timestamp"].dt.month.values

    # ── Sample weights from clearsky GHI ────────────────────────────────────
    if WEIGHT_COL not in train.columns:
        raise SystemExit(f"Required weighting column '{WEIGHT_COL}' missing from train.")
    weights = train[WEIGHT_COL].values.astype(np.float64) / WEIGHT_SCALE
    weights = np.clip(weights, WEIGHT_FLOOR, 1.0)
    log.info(f"Sample-weight summary (from {WEIGHT_COL} / {WEIGHT_SCALE:g}, "
             f"floor={WEIGHT_FLOOR}):")
    log.info(f"  mean={weights.mean():.3f}  median={np.median(weights):.3f}  "
             f"max={weights.max():.3f}  effective n={weights.sum():.0f} / {len(weights)}")
    elev_train = train["ext_sol_elevation"].values
    for lo, hi, name in [(-90, 0, "night"), (0, 15, "twilight"), (15, 30, "low"),
                          (30, 45, "mid"), (45, 90, "high")]:
        m = (elev_train > lo) & (elev_train <= hi)
        if m.any():
            log.info(f"  bucket {name:8s} (n={m.sum():>7,d}): "
                     f"mean weight={weights[m].mean():.3f}")

    params = dict(
        objective="regression", metric="rmse",
        learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
        verbose=-1, n_jobs=-1, seed=42,
    )

    log.info("Running 6-fold GroupKFold (by month) CV — clearsky-weighted ...")
    fold_scores = []
    oof_pred = np.zeros(len(train), dtype=np.float64)
    gkf = GroupKFold(n_splits=6)
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        ds_tr = lgb.Dataset(X_train.iloc[tr_idx], label=y_train[tr_idx],
                            weight=weights[tr_idx],
                            categorical_feature=CATEGORICAL_FEATURES)
        # Validation uses uniform weight so early stopping tracks raw RMSE
        # — the actual leaderboard metric, not the weighted training loss.
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
        del ds_tr, ds_va, model, pred
        gc.collect()

    mbe_all, rmse_all, score_all = score_components(y_train, oof_pred)
    log.info(f"OOF (unweighted, full dataset):  "
             f"|MBE|={mbe_all:.2f}  RMSE={rmse_all:.2f}  score={score_all:.2f}")
    log.info("  v10 baseline OOF for comparison: |MBE|≈2.7  RMSE≈72  score=37.91")

    # Regime-bucket diagnostic — does the high-sun bucket actually improve?
    log.info("OOF by regime (using ext_sol_elevation bins):")
    for lo, hi, name in [(-90, 0, "night"), (0, 15, "twilight"), (15, 30, "low"),
                          (30, 45, "mid"), (45, 90, "high")]:
        m = (elev_train > lo) & (elev_train <= hi)
        if not m.any():
            continue
        mbe_b, rmse_b, _ = score_components(y_train[m], oof_pred[m])
        log.info(f"  {name:8s} ({lo:+3d}<elev≤{hi:+3d}, n={m.sum():>7,d}):  "
                 f"|MBE|={mbe_b:6.2f}  RMSE={rmse_b:7.2f}")

    # Save OOF for stacking
    oof_df = pd.DataFrame({
        ID: train[ID].values,
        "TargetMBE":  np.clip(oof_pred, 0, 1361),
        "TargetRMSE": np.clip(oof_pred, 0, 1361),
    })
    oof_df.to_csv(SUBS / f"{RUN_TAG}_oof.csv", index=False)
    log.info(f"Saved OOF preds: {SUBS / f'{RUN_TAG}_oof.csv'}")

    best_iter = int(np.mean([fs["best_iter"] for fs in fold_scores]))
    log.info(f"Refitting on full train for {best_iter} rounds ...")
    full_ds = lgb.Dataset(X_train, label=y_train, weight=weights,
                          categorical_feature=CATEGORICAL_FEATURES)
    final_model = lgb.train(params, full_ds, num_boost_round=best_iter)

    log.info("Predicting test ...")
    test_pred = final_model.predict(X_test)

    # Per-(station, hour) nighttime override
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
    log.info(f"  rows overridden as night: {int(is_night.sum()):,} ({is_night.mean()*100:.1f}%)")
    test_pred = np.clip(test_pred, 0, 1361)

    sub = pd.DataFrame({ID: test[ID], "TargetMBE": test_pred, "TargetRMSE": test_pred})
    sample = pd.read_csv(SAMP)
    sub = sample[[ID]].merge(sub, on=ID, how="left")
    if sub[["TargetMBE", "TargetRMSE"]].isna().any().any():
        log.error("  predictions missing — check ID alignment")

    sub_path = SUBS / f"{RUN_TAG}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}  ({len(sub):,} rows)")

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
        "weight_col":       WEIGHT_COL,
        "weight_scale":     WEIGHT_SCALE,
        "weight_floor":     WEIGHT_FLOOR,
        "weight_stats":     {
            "mean":   round(float(weights.mean()), 3),
            "median": round(float(np.median(weights)), 3),
            "max":    round(float(weights.max()), 3),
            "effective_n": round(float(weights.sum()), 0),
        },
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
