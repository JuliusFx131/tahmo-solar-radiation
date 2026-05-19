"""
TAHMO Solar Radiation — LGBM v14, iterated pseudo-labels.

v12 was trained from a v10/v11/v8 consensus and became the new LB best
(32.62). v14 takes v12 as the new anchor and rebuilds the pseudo-label
consensus from (v12, v10, v11) — dropping v8 which is now too weak and
would dilute the consensus std.

Everything else is held constant vs v12 (std<8 threshold, weight=0.5,
same 179-feature set, same memory-safe LGBM params) so the LB delta is a
clean ablation of the consensus source.

Run:
  bash /workspace/shell/run_train_v14_pseudo_v12.sh

Output:
  submissions/lgbm_v14_pseudo_v12.csv
  submissions/lgbm_v14_pseudo_v12_log.txt
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
NIGHT = PROC / "night_offset_per_station.csv"
SAMP  = ROOT / "data" / "raw" / "SampleSubmission.csv"

# Iterated consensus: v12 (current LB best, 32.62) anchors, v10 and v11
# add diversity. v8 dropped — too weak now, would inflate std and exclude
# good pseudos.
SOURCE_SUBS = [
    SUBS / "lgbm_v12_pseudo_v10.csv",
    SUBS / "lgbm_v10_csr.csv",
    SUBS / "lgbm_v11_m2.csv",
]

# Pseudo-labeling knobs
PSEUDO_STD_THRESH = 8.0         # W/m² — very tight (memory headroom for 8 GB cgroup)
PSEUDO_WEIGHT     = 0.5
DAYTIME_ELEV_THR  = 5.0

TARGET = "radiation (W/m2)"
ID     = "ID"
RUN_TAG = "lgbm_v14_pseudo_v12"


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
    """Memory-efficient per-station rolling/lag/diff features (same as v8)."""
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


def build_pseudo_labels(test, source_subs):
    """Pick high-confidence test rows: low std across models + daytime."""
    preds = []
    for path in source_subs:
        if not path.exists():
            log.warning(f"  {path.name} missing — pseudo-label confidence proxy will use fewer sources")
            continue
        sub = pd.read_csv(path).set_index("ID")["TargetMBE"]
        preds.append(sub)
    if len(preds) < 2:
        raise RuntimeError("Need at least 2 source submissions for confidence proxy")

    pred_df = pd.concat(preds, axis=1, keys=[p.stem for p in source_subs[:len(preds)]])
    pred_df.columns = [f"pred_{c}" for c in pred_df.columns.get_level_values(0)]
    pred_df["pred_mean"] = pred_df.mean(axis=1)
    pred_df["pred_std"]  = pred_df.std(axis=1)
    pred_df = pred_df.reset_index()

    # Join onto test for elevation context
    merged = test[["ID", "ext_sol_elevation"]].merge(pred_df, on="ID", how="left")

    confident = (
        (merged["pred_std"] < PSEUDO_STD_THRESH)
        & (merged["ext_sol_elevation"] > DAYTIME_ELEV_THR)
        & merged["pred_mean"].notna()
    )
    log.info(f"  pseudo-label candidates (daytime + std < {PSEUDO_STD_THRESH}): "
             f"{confident.sum():,} / {len(merged):,} test rows ({confident.mean()*100:.1f}%)")

    pseudo = merged.loc[confident, ["ID", "pred_mean"]].rename(columns={"pred_mean": TARGET})
    return pseudo


def main():
    t0 = time.time()
    log.info("Loading enhanced train + test ...")
    train = pd.read_csv(TRAIN, parse_dates=["timestamp"])
    test  = pd.read_csv(TEST,  parse_dates=["timestamp"])
    log.info(f"  train: {train.shape}   test: {test.shape}")

    log.info("Building pseudo-label set from existing submissions ...")
    pseudo_targets = build_pseudo_labels(test, SOURCE_SUBS)
    log.info(f"  selected {len(pseudo_targets):,} pseudo-label rows from test")

    # Get the test rows that match pseudo IDs (need full feature columns)
    pseudo_test = test[test["ID"].isin(pseudo_targets["ID"])].copy()
    pseudo_test = pseudo_test.merge(pseudo_targets, on="ID", how="left", suffixes=("", "_pseudo"))
    pseudo_test[TARGET] = pseudo_test[f"{TARGET}_pseudo"] if f"{TARGET}_pseudo" in pseudo_test.columns else pseudo_test[TARGET]
    if f"{TARGET}_pseudo" in pseudo_test.columns:
        pseudo_test = pseudo_test.drop(columns=[f"{TARGET}_pseudo"])
    # Tag origin so we can apply sample weights
    train["_origin"] = "train"
    pseudo_test["_origin"] = "pseudo"
    test["_origin"] = "test"

    # Augmented training set = original train + pseudo-labeled test
    aug_train = pd.concat([train, pseudo_test], ignore_index=True)
    log.info(f"  augmented training set: {len(train):,} → {len(aug_train):,} rows "
             f"(+{len(pseudo_test):,} pseudo)")

    # Downcast & feature engineering (same as v8)
    log.info("Downcasting + feature engineering ...")
    for df in (aug_train, test):
        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = df[c].astype(np.float32)

    aug_train = add_time_features(ffill_cams(aug_train))
    aug_train = add_rolling_features(aug_train)
    test      = add_time_features(ffill_cams(test))
    test      = add_rolling_features(test)

    # Categoricals
    for col in ["station", "country"]:
        cats = sorted(set(aug_train[col].astype(str)) | set(test[col].astype(str)))
        aug_train[col] = pd.Categorical(aug_train[col].astype(str), categories=cats)
        test[col]      = pd.Categorical(test[col].astype(str), categories=cats)

    # Feature list (same as v8)
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
        # v7 forward weather
        "ext_fw_temp_lead_15m", "ext_fw_temp_lead_30m",
        "ext_fw_temp_lead_1h",  "ext_fw_temp_lead_3h",
        "ext_fw_temp_diff_1h",  "ext_fw_temp_diff_3h",
        "ext_fw_rh_lead_15m",   "ext_fw_rh_lead_30m",
        "ext_fw_rh_lead_1h",    "ext_fw_rh_lead_3h",
        "ext_fw_rh_diff_1h",
        "ext_fw_precip_lead_15m", "ext_fw_precip_lead_30m",
        "ext_fw_precip_lead_1h",  "ext_fw_precip_lead_3h",
        "ext_fw_precip_sum_lead_3h", "ext_fw_precip_sum_lead_24h",
        # v8 daily aggregates
        "ext_dd_temp_max", "ext_dd_temp_min", "ext_dd_temp_amp",
        "ext_dd_temp_mean", "ext_dd_temp_std",
        "ext_dd_rh_max", "ext_dd_rh_min", "ext_dd_rh_mean",
        "ext_dd_precip_sum", "ext_dd_precip_max",
        "ext_dd_om_ghi_max", "ext_dd_om_ghi_sum", "ext_dd_om_cc_mean",
        "ext_dd_np_ghi_max", "ext_dd_np_ghi_sum",
        "ext_dd_ssrd_max",   "ext_dd_ssrd_sum",
        # v8 energy balance
        "ext_eb_dT_x_blh_1h", "ext_eb_dT_x_blh_3h",
        "ext_eb_dRH_x_blh_1h",
        "ext_eb_T_minus_dewpoint", "ext_eb_water_content_proxy",
        "ext_eb_dT_per_W_per_m2",
        "ext_eb_morning_warming", "ext_eb_afternoon_cooling",
        # v10: CAMS Solar Radiation Timeseries (the headline feature)
        "ext_csr_ghi", "ext_csr_bhi", "ext_csr_dhi", "ext_csr_bni",
        "ext_csr_clearsky_ghi", "ext_csr_clearsky_bhi",
        "ext_csr_clearsky_dhi", "ext_csr_clearsky_bni",
        "ext_csr_reliability",
        "ext_csr_clearness_kt", "ext_csr_diffuse_fraction",
        # v11: MERRA-2 speciated aerosols
        "ext_m2_aod_total", "ext_m2_aod_dust", "ext_m2_aod_oc", "ext_m2_aod_bc",
        "ext_m2_aod_ss", "ext_m2_angstrom",
        "ext_m2_pm25_dust", "ext_m2_pm25_oc",
    ]
    TIME = ["hour", "minute", "month", "doy", "year", "dow", "dom",
            "hour_sin", "hour_cos", "doy_sin", "doy_cos",
            "solar_clock", "hours_from_solar_noon", "days_since_install"]
    ROLLS = [c for c in aug_train.columns
             if any(c.endswith(suf) for suf in
                    ["_mean_1h", "_mean_3h", "_mean_6h", "_mean_24h",
                     "_lag_1h", "_lag_3h", "_diff_1h", "_sum_6h", "_sum_24h"])
             and not c.startswith("ext_fw_") and not c.startswith("ext_eb_")]
    CATEGORICAL_FEATURES = ["station", "country"]
    FEATURES = list(dict.fromkeys(BASE + TIME + ROLLS + CATEGORICAL_FEATURES))
    log.info(f"  features used: {len(FEATURES)}  ({len(ROLLS)} rolling/lag)")

    X_train = aug_train[FEATURES].copy()
    y_train = aug_train[TARGET].values
    weights = np.where(aug_train["_origin"] == "pseudo",
                       np.float32(PSEUDO_WEIGHT), np.float32(1.0))
    X_test  = test[FEATURES].copy()
    groups  = aug_train["timestamp"].dt.month.values
    origin_arr = aug_train["_origin"].values.copy()  # for OOF filtering later
    n_train_aug = len(aug_train)                     # cache before deletion

    # Stash the test columns we still need for night override + submission build
    test_meta = test[[ID, "station", "timestamp", "ext_sol_elevation"]].copy()

    # Free large intermediate frames before lgb.Dataset construction (8 GB cgroup)
    del aug_train, test
    gc.collect()

    params = dict(
        objective="regression", metric="rmse",
        learning_rate=0.05, num_leaves=47, min_data_in_leaf=120,    # smaller model = less memory
        feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
        verbose=-1, n_jobs=-1, seed=42,
        histogram_pool_size=512,                                      # cap LGBM's hist pool (MB)
    )

    # CV by month — note: pseudo rows are even months. They'll be split across
    # folds purely by their (faked) month. Their inclusion as TRAINING during
    # folds that hold out odd months still gives the model extra data without
    # leakage of the val month's true radiation.
    log.info("Running 6-fold GroupKFold (by month) CV on augmented data ...")
    fold_scores = []
    oof_pred = np.zeros(n_train_aug)
    gkf = GroupKFold(n_splits=6)
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        ds_tr = lgb.Dataset(X_train.iloc[tr_idx], label=y_train[tr_idx],
                            weight=weights[tr_idx],
                            categorical_feature=CATEGORICAL_FEATURES)
        ds_va = lgb.Dataset(X_train.iloc[va_idx], label=y_train[va_idx],
                            weight=weights[va_idx],
                            categorical_feature=CATEGORICAL_FEATURES)
        model = lgb.train(params, ds_tr, num_boost_round=2000, valid_sets=[ds_va],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        pred = model.predict(X_train.iloc[va_idx])
        oof_pred[va_idx] = pred
        held_out_month = groups[va_idx][0]
        # Score on ORIGINAL train rows only (not on pseudo), for honest comparison
        orig_mask_va = (origin_arr[va_idx] == "train")
        if orig_mask_va.sum() > 0:
            mbe, rmse, score = score_components(
                y_train[va_idx][orig_mask_va], pred[orig_mask_va])
            fold_scores.append({"fold": fold_idx, "month_held_out": int(held_out_month),
                                "mbe": mbe, "rmse": rmse, "score": score,
                                "n_orig": int(orig_mask_va.sum()),
                                "best_iter": model.best_iteration})
            log.info(f"  fold {fold_idx} (month {held_out_month}, original-only):  "
                     f"|MBE|={mbe:6.2f}  RMSE={rmse:6.2f}  score={score:6.2f}  "
                     f"best={model.best_iteration}")
        del ds_tr, ds_va, model, pred
        gc.collect()

    # OOF over original-train rows only
    orig_all = (origin_arr == "train")
    mbe_all, rmse_all, score_all = score_components(y_train[orig_all], oof_pred[orig_all])
    log.info(f"OOF (original train only): |MBE|={mbe_all:.2f}  RMSE={rmse_all:.2f}  "
             f"score={score_all:.2f}")

    best_iter = int(np.mean([fs["best_iter"] for fs in fold_scores]))
    log.info(f"Refitting on full augmented train for {best_iter} rounds ...")
    full_ds = lgb.Dataset(X_train, label=y_train, weight=weights,
                          categorical_feature=CATEGORICAL_FEATURES)
    final = lgb.train(params, full_ds, num_boost_round=best_iter)

    log.info("Predicting test ...")
    test_pred = final.predict(X_test)

    # Per-(station, hour) night override
    log.info("Applying per-(station, hour) night override ...")
    train_night_src = pd.read_csv(TRAIN,
                                  usecols=["station", "timestamp", "ext_sol_elevation", TARGET],
                                  parse_dates=["timestamp"])
    train_night_src = train_night_src[train_night_src["ext_sol_elevation"] <= 0].copy()
    train_night_src["hour"] = train_night_src["timestamp"].dt.hour
    per_hour_mean    = train_night_src.groupby(["station", "hour"])[TARGET].mean().to_dict()
    per_station_mean = train_night_src.groupby("station")[TARGET].mean().to_dict()
    fallback = float(np.mean(list(per_station_mean.values())) if per_station_mean else 0.0)

    is_night = test_meta["ext_sol_elevation"] <= 0
    test_hours = test_meta["timestamp"].dt.hour.values
    test_stations = test_meta["station"].astype(str).values
    night_vals = np.array([per_hour_mean.get((s, h), per_station_mean.get(s, fallback))
                           for s, h in zip(test_stations, test_hours)])
    test_pred = np.where(is_night, night_vals, test_pred)
    test_pred = np.clip(test_pred, 0, 1361)

    sub = pd.DataFrame({ID: test_meta[ID], "TargetMBE": test_pred, "TargetRMSE": test_pred})
    sample = pd.read_csv(SAMP)
    sub = sample[[ID]].merge(sub, on=ID, how="left")

    sub_path = SUBS / f"{RUN_TAG}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}  ({len(sub):,} rows, "
             f"mean={test_pred.mean():.1f})")

    log_path = SUBS / f"{RUN_TAG}_log.txt"
    log_path.write_text(json.dumps({
        "tag":               RUN_TAG,
        "source_subs":       [str(p) for p in SOURCE_SUBS],
        "pseudo_std_thresh": PSEUDO_STD_THRESH,
        "pseudo_weight":     PSEUDO_WEIGHT,
        "n_pseudo_rows":     int(len(pseudo_test)),
        "n_features":        len(FEATURES),
        "cv_folds":          fold_scores,
        "cv_oof_score":      round(score_all, 3),
        "cv_oof_mbe":        round(mbe_all, 3),
        "cv_oof_rmse":       round(rmse_all, 3),
        "best_iter":         best_iter,
        "submission":        str(sub_path),
        "wall_time_s":       round(time.time() - t0, 1),
    }, indent=2, default=str))
    log.info(f"Wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
