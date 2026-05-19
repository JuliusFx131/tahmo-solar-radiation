"""
TAHMO Solar Radiation — v15 soft-gated ensemble
================================================
Two LGBMs on the v10 feature set, sample-weighted by solar elevation so each
specialises in a different regime, then blended with a smooth elevation gate.

Motivation: a single LGBM trained with squared-error loss splits its capacity
across all sun angles uniformly. The leaderboard RMSE is dominated by absolute
errors at high radiation (noon, clear sky), but the model also has to handle
dawn/dusk/cloudy-low rows where the radiation level itself is small. By
training one expert per regime with sample weights, each can specialise
without fighting for the same splits.

  Model A (low-sun expert):  weight = sigmoid((30 - elev) / 10)
  Model B (high-sun expert): weight = 1 - weight_A
  Gate at inference:         g = sigmoid((elev - 30) / 10)
  Blended prediction:        (1 - g) * pred_A + g * pred_B

Threshold = 30° (rough boundary between high-airmass dawn/dusk and noon-ish
high-sun regime). Width = 10° gives a smooth transition over elev ∈ [20°, 40°].

Night rows (elev <= 0) still use the per-(station, hour) override from the
training set.

Run:
  bash /workspace/shell/run_train_v15_gated.sh

Outputs:
  /workspace/submissions/lgbm_v15_gated.csv
  /workspace/submissions/lgbm_v15_gated_log.txt
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
RUN_TAG = "lgbm_v15_gated"

CATEGORICAL_FEATURES = ["station", "country"]

# Soft-gate parameters
GATE_ELEV_MID   = 30.0   # degrees — midpoint of transition
GATE_ELEV_WIDTH = 10.0   # degrees — softness


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


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


def train_one_expert(
    tag: str,
    X_train: pd.DataFrame, y_train: np.ndarray, weights: np.ndarray, groups: np.ndarray,
    X_test: pd.DataFrame, params: dict,
):
    """Run 6-fold GroupKFold CV with sample weights, return OOF preds and a
    refit-on-full model's test predictions, plus the per-fold scores."""
    log.info(f"[{tag}] sample-weight summary: "
             f"mean={weights.mean():.3f}  median={np.median(weights):.3f}  "
             f"effective n={weights.sum():.0f}")
    fold_scores = []
    oof_pred = np.zeros(len(X_train), dtype=np.float64)
    gkf = GroupKFold(n_splits=6)
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        ds_tr = lgb.Dataset(X_train.iloc[tr_idx], label=y_train[tr_idx],
                            weight=weights[tr_idx],
                            categorical_feature=CATEGORICAL_FEATURES)
        # IMPORTANT: validation uses uniform weight so early-stopping tracks raw RMSE
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
        log.info(f"  [{tag}] fold {fold_idx} (month {held_out_month}): "
                 f"|MBE|={mbe:6.2f}  RMSE={rmse:6.2f}  score={score:6.2f}  "
                 f"best_iter={model.best_iteration}")
        del ds_tr, ds_va, model, pred
        gc.collect()

    best_iter = int(np.mean([fs["best_iter"] for fs in fold_scores]))
    log.info(f"[{tag}] refit on full train for {best_iter} rounds ...")
    full_ds = lgb.Dataset(X_train, label=y_train, weight=weights,
                          categorical_feature=CATEGORICAL_FEATURES)
    final_model = lgb.train(params, full_ds, num_boost_round=best_iter)
    test_pred = final_model.predict(X_test)
    log.info(f"[{tag}] top-10 features by gain:")
    imp = pd.DataFrame({
        "feature":   final_model.feature_name(),
        "gain":      final_model.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    for _, row in imp.head(10).iterrows():
        log.info(f"  [{tag}]  {row['feature']:40s}  gain={row['gain']:>12,.0f}")
    del full_ds, final_model
    gc.collect()
    return oof_pred, test_pred, fold_scores, imp.head(15).to_dict("records")


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
    # Drop any BASE features the dataframe doesn't actually have (defensive)
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

    # ── Soft-gate weights, computed from ext_sol_elevation ──────────────────
    elev_train = train["ext_sol_elevation"].values.astype(np.float64)
    elev_test  = test ["ext_sol_elevation"].values.astype(np.float64)

    # Model A specialises in low-sun (dawn/dusk/cloudy + nighttime)
    w_A = sigmoid((GATE_ELEV_MID - elev_train) / GATE_ELEV_WIDTH)
    # Model B specialises in high-sun (noon-ish, clear-sky daytime)
    w_B = 1.0 - w_A

    # Gate for inference (matches Model B's domain)
    g_test = sigmoid((elev_test - GATE_ELEV_MID) / GATE_ELEV_WIDTH)

    log.info(f"Gate sanity: train rows above elev=30°: "
             f"{(elev_train > 30).mean()*100:.1f}%   "
             f"test rows above elev=30°: {(elev_test > 30).mean()*100:.1f}%")

    params = dict(
        objective="regression", metric="rmse",
        learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
        verbose=-1, n_jobs=-1, seed=42,
    )

    # ── Expert A (low-sun) ──────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("Training Expert A (LOW-SUN) — weights peak at elev=0")
    log.info("=" * 70)
    oof_A, test_A, folds_A, imp_A = train_one_expert(
        "A_low", X_train, y_train, w_A, groups, X_test, params,
    )

    # ── Expert B (high-sun) ─────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("Training Expert B (HIGH-SUN) — weights peak at elev=60")
    log.info("=" * 70)
    oof_B, test_B, folds_B, imp_B = train_one_expert(
        "B_high", X_train, y_train, w_B, groups, X_test, params,
    )

    # ── OOF scoring: individual + gated blend ──────────────────────────────
    g_train = sigmoid((elev_train - GATE_ELEV_MID) / GATE_ELEV_WIDTH)
    oof_blend = (1.0 - g_train) * oof_A + g_train * oof_B

    mbe_A, rmse_A, score_A = score_components(y_train, oof_A)
    mbe_B, rmse_B, score_B = score_components(y_train, oof_B)
    mbe_E, rmse_E, score_E = score_components(y_train, oof_blend)

    log.info("=" * 70)
    log.info("OOF scores (full dataset, includes night rows):")
    log.info(f"  Expert A (low-sun):  |MBE|={mbe_A:6.2f}  RMSE={rmse_A:6.2f}  score={score_A:6.2f}")
    log.info(f"  Expert B (high-sun): |MBE|={mbe_B:6.2f}  RMSE={rmse_B:6.2f}  score={score_B:6.2f}")
    log.info(f"  Gated blend:         |MBE|={mbe_E:6.2f}  RMSE={rmse_E:6.2f}  score={score_E:6.2f}")

    # Regime-specific OOF breakdown so we can see where each expert wins
    log.info("OOF by regime (using ext_sol_elevation bins):")
    bins = [(-90, 0, "night"), (0, 15, "twilight"), (15, 30, "low"),
            (30, 45, "mid"), (45, 90, "high")]
    for lo, hi, name in bins:
        m = (elev_train > lo) & (elev_train <= hi)
        if not m.any():
            continue
        _, rmse_A_b, _ = score_components(y_train[m], oof_A[m])
        _, rmse_B_b, _ = score_components(y_train[m], oof_B[m])
        _, rmse_E_b, _ = score_components(y_train[m], oof_blend[m])
        log.info(f"  {name:8s} ({lo:+3d}<elev≤{hi:+3d}, n={m.sum():>7,d}):  "
                 f"RMSE  A={rmse_A_b:6.2f}  B={rmse_B_b:6.2f}  blend={rmse_E_b:6.2f}")

    # ── Build the test submission ──────────────────────────────────────────
    test_pred = (1.0 - g_test) * test_A + g_test * test_B

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

    # Also save the unblended OOFs for inspection / future stacking
    oof_df = pd.DataFrame({
        ID: train[ID].values,
        "elev": elev_train,
        "y": y_train,
        "oof_A_low":  oof_A,
        "oof_B_high": oof_B,
        "oof_blend":  oof_blend,
    })
    oof_df.to_csv(SUBS / f"{RUN_TAG}_oof.csv", index=False)
    log.info(f"Saved OOF preds: {SUBS / f'{RUN_TAG}_oof.csv'}")

    log_path = SUBS / f"{RUN_TAG}_log.txt"
    info = {
        "tag":              RUN_TAG,
        "wall_time_s":      round(time.time() - t0, 1),
        "n_train":          int(len(train)),
        "n_test":           int(len(test)),
        "n_features":       len(FEATURES),
        "gate_elev_mid":    GATE_ELEV_MID,
        "gate_elev_width":  GATE_ELEV_WIDTH,
        "lgbm_params":      params,
        "expert_A_folds":   folds_A,
        "expert_B_folds":   folds_B,
        "oof_A":     {"mbe": round(mbe_A, 3), "rmse": round(rmse_A, 3), "score": round(score_A, 3)},
        "oof_B":     {"mbe": round(mbe_B, 3), "rmse": round(rmse_B, 3), "score": round(score_B, 3)},
        "oof_blend": {"mbe": round(mbe_E, 3), "rmse": round(rmse_E, 3), "score": round(score_E, 3)},
        "submission":       str(sub_path),
        "feature_importance_top15_A": imp_A,
        "feature_importance_top15_B": imp_B,
    }
    log_path.write_text(json.dumps(info, indent=2, default=str))
    log.info(f"Saved run log:   {log_path}")
    log.info(f"Wall time: {info['wall_time_s']}s")


if __name__ == "__main__":
    main()
