"""
Per-station LGBM. Trains 40 separate models — one per station — using the
v4 feature set (the proven LB winner). Each station gets its own model with
reduced complexity (fewer leaves, more regularisation) suited to its smaller
training set (~16k rows per station).

Rationale: the shared model has to compromise across 40 stations with
different sensor characteristics, climates, drift patterns. Per-station can
specialise to each.

Avoids temporal-neighbour features (those caused a LB disaster in v6 due to
test-time distribution shift).

Uses v4 feature set:
  • Solar geometry, pvlib clear-sky, raw weather, ERA5, CAMS (7 species)
  • Open-Meteo, NASA POWER
  • Rolling/lag of temp/rh/precip/ssrd/tcc/om_ghi/np_ghi
  • Time features (hour, doy, etc.)
  • Per-(station, hour) night override at inference

Run:
  bash /workspace/shell/run_train_per_station.sh

Output:
  submissions/lgbm_per_station.csv
  submissions/lgbm_per_station_log.txt
"""

import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

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
RUN_TAG = "lgbm_per_station"


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


def main():
    t0 = time.time()
    log.info("Loading data ...")
    train = pd.read_csv(TRAIN, parse_dates=["timestamp"])
    test  = pd.read_csv(TEST,  parse_dates=["timestamp"])
    log.info(f"  train: {train.shape}  test: {test.shape}")

    # Downcast & basic features (no rolling — keeps memory bounded per-station)
    for df in (train, test):
        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = df[c].astype("float32")

    train = add_time_features(ffill_cams(train))
    test  = add_time_features(ffill_cams(test))

    # Feature list — v4-style without temporal_neighbors (those broke v6).
    BASE = [
        "precipitation (mm)", "relativehumidity (-)", "temperature (degrees Celsius)",
        "installation_height", "elevation",
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
    ]
    TIME = ["hour", "minute", "month", "doy", "year", "dow", "dom",
            "hour_sin", "hour_cos", "doy_sin", "doy_cos",
            "solar_clock", "hours_from_solar_noon", "days_since_install"]
    FEATURES = [c for c in BASE + TIME if c in train.columns]
    log.info(f"  features per-station: {len(FEATURES)}")

    # Per-station LGBM params — lower complexity than the shared model.
    params = dict(
        objective="regression", metric="rmse",
        learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
        verbose=-1, n_jobs=-1, seed=42,
    )

    test_preds = pd.Series(np.nan, index=test.index, dtype=np.float64)
    stations = sorted(train["station"].unique())
    fold_scores = []

    for sta_idx, sta in enumerate(stations, 1):
        tr_sta = train[train["station"] == sta]
        te_sta = test[test["station"] == sta]
        if len(tr_sta) < 100 or len(te_sta) == 0:
            log.warning(f"  [{sta_idx:>2}/{len(stations)}] {sta} skipped "
                        f"(train={len(tr_sta)}, test={len(te_sta)})")
            continue

        # Last 10% of training rows as holdout for early stopping
        n_holdout = max(500, int(0.10 * len(tr_sta)))
        tr_main = tr_sta.iloc[:-n_holdout]
        tr_val  = tr_sta.iloc[-n_holdout:]

        X_tr, y_tr = tr_main[FEATURES], tr_main[TARGET].values
        X_va, y_va = tr_val[FEATURES],  tr_val[TARGET].values
        X_te       = te_sta[FEATURES]

        ds_tr = lgb.Dataset(X_tr, label=y_tr)
        ds_va = lgb.Dataset(X_va, label=y_va)
        model = lgb.train(params, ds_tr, num_boost_round=2000, valid_sets=[ds_va],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        val_pred = model.predict(X_va)
        mbe, rmse, score = score_components(y_va, val_pred)
        fold_scores.append({"station": sta, "n_train": len(tr_sta),
                            "mbe": mbe, "rmse": rmse, "score": score,
                            "best_iter": model.best_iteration})
        log.info(f"  [{sta_idx:>2}/{len(stations)}] {sta}  "
                 f"n_train={len(tr_sta):>6}  |MBE|={mbe:6.2f}  RMSE={rmse:6.2f}  "
                 f"score={score:6.2f}  best={model.best_iteration}")

        # Refit on FULL station training data for the final test prediction
        full_ds = lgb.Dataset(tr_sta[FEATURES], label=tr_sta[TARGET].values)
        final = lgb.train(params, full_ds, num_boost_round=model.best_iteration)
        test_preds.iloc[te_sta.index] = final.predict(X_te)

        del ds_tr, ds_va, model, full_ds, final, X_tr, y_tr, X_va, y_va, X_te
        gc.collect()

    # Per-(station, hour) night override
    log.info("Applying per-(station, hour) night override ...")
    train_night = train[train["ext_sol_elevation"] <= 0].copy()
    train_night["hour"] = train_night["timestamp"].dt.hour
    per_hour_mean    = train_night.groupby(["station", "hour"])[TARGET].mean().to_dict()
    per_station_mean = train_night.groupby("station")[TARGET].mean().to_dict()
    fallback = float(np.mean(list(per_station_mean.values())) if per_station_mean else 0.0)

    is_night = test["ext_sol_elevation"] <= 0
    test_hours = test["timestamp"].dt.hour.values
    test_stations = test["station"].astype(str).values
    night_vals = np.array([per_hour_mean.get((s, h), per_station_mean.get(s, fallback))
                           for s, h in zip(test_stations, test_hours)])
    test_pred_arr = test_preds.values.copy()
    test_pred_arr = np.where(is_night, night_vals, test_pred_arr)
    test_pred_arr = np.clip(test_pred_arr, 0, 1361)

    # Compose submission
    sub = pd.DataFrame({ID: test[ID], "TargetMBE": test_pred_arr, "TargetRMSE": test_pred_arr})
    sample = pd.read_csv(SAMP)
    sub = sample[[ID]].merge(sub, on=ID, how="left")
    if sub[["TargetMBE", "TargetRMSE"]].isna().any().any():
        log.error("  predictions missing — check ID alignment")

    sub_path = SUBS / f"{RUN_TAG}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}  ({len(sub):,} rows, "
             f"mean={test_pred_arr.mean():.1f})")

    avg_mbe  = np.mean([fs["mbe"]  for fs in fold_scores])
    avg_rmse = np.mean([fs["rmse"] for fs in fold_scores])
    avg_score= np.mean([fs["score"] for fs in fold_scores])
    log.info(f"Holdout-avg across stations:  |MBE|={avg_mbe:.2f}  "
             f"RMSE={avg_rmse:.2f}  score={avg_score:.2f}")

    log_path = SUBS / f"{RUN_TAG}_log.txt"
    log_path.write_text(json.dumps({
        "tag":              RUN_TAG,
        "n_stations":       len(fold_scores),
        "features":         FEATURES,
        "params":           params,
        "per_station":      fold_scores,
        "avg_mbe":          avg_mbe,
        "avg_rmse":         avg_rmse,
        "avg_score":        avg_score,
        "submission":       str(sub_path),
        "wall_time_s":      round(time.time() - t0, 1),
    }, indent=2, default=str))
    log.info(f"Wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
