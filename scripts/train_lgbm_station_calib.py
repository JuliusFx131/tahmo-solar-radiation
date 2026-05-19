"""
TAHMO Solar Radiation — LGBM + per-station calibration (v1)
============================================================
Same shared LGBM as the raw baseline, plus a per-station linear correction
fit on out-of-fold daytime predictions:

    radiation_obs ≈ a_s + b_s * radiation_pred           (per station s)

Rationale (forum thread, 11 May 2026): the LB rewards reproducing each
sensor's idiosyncratic offset + scale. The shared model captures the
physical signal; (a_s, b_s) lets each station retain its sensor-specific
bias and gain.

Pipeline:
  1. 6-fold GroupKFold-by-month CV → OOF predictions for every training row.
  2. Fit (a_s, b_s) per station on daytime OOF rows. Fall back to identity
     for stations with too few rows or wild coefficients.
  3. Refit shared LGBM on all training data.
  4. Predict test → apply (a_s, b_s) per station → night override → clip.

Run:
  bash /workspace/shell/run_train_lgbm_station_calib.sh

Outputs:
  /workspace/submissions/lgbm_station_calib_v1.csv
  /workspace/submissions/lgbm_station_calib_v1_log.txt
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
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
RUN_TAG = "lgbm_station_calib_v1"

# Calibration safety: stations with fewer daytime rows than this fall back
# to identity (a=0, b=1). And we reject any fit with extreme coefficients.
MIN_CALIB_ROWS = 500
B_MIN, B_MAX = 0.5, 1.5  # acceptable slope range
A_MAX_ABS = 200          # acceptable intercept magnitude (W/m²)
DAY_ELEV_THRESHOLD = 5.0 # only fit on rows where the sun is well above horizon

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


def fit_station_calibration(df_train, oof_pred):
    """Return {station: (a, b, n_rows, used_calib)} from daytime OOF rows."""
    calib = {}
    df = df_train.copy()
    df["oof"] = oof_pred
    daytime_mask = df["ext_sol_elevation"] > DAY_ELEV_THRESHOLD
    df = df[daytime_mask]

    lr = LinearRegression()
    for station, g in df.groupby("station", observed=True):
        n = len(g)
        if n < MIN_CALIB_ROWS:
            calib[str(station)] = (0.0, 1.0, n, False)
            continue
        X = g["oof"].values.reshape(-1, 1)
        y = g[TARGET].values
        lr.fit(X, y)
        a = float(lr.intercept_)
        b = float(lr.coef_[0])
        used = (B_MIN <= b <= B_MAX) and (abs(a) <= A_MAX_ABS)
        if not used:
            a, b = 0.0, 1.0
        calib[str(station)] = (a, b, n, used)
    return calib


def apply_station_calibration(df, pred, calib):
    """Apply (a, b) per station — only on daytime rows."""
    pred = pred.copy()
    daytime_mask = df["ext_sol_elevation"] > DAY_ELEV_THRESHOLD
    for station in df["station"].cat.categories:
        a, b, _, used = calib.get(str(station), (0.0, 1.0, 0, False))
        if not used:
            continue
        mask = daytime_mask & (df["station"] == station)
        idx = np.where(mask.values)[0]
        if len(idx):
            pred[idx] = a + b * pred[idx]
    return pred


def main():
    t0 = time.time()
    log.info("Loading data ...")
    train = pd.read_csv(TRAIN, parse_dates=["timestamp"])
    test  = pd.read_csv(TEST,  parse_dates=["timestamp"])
    log.info(f"  train: {train.shape}   test: {test.shape}")

    train = add_time_features(ffill_cadence_features(train))
    test  = add_time_features(ffill_cadence_features(test))

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

    mbe_raw, rmse_raw, score_raw = score_components(y_train, oof_pred)
    log.info(f"OOF raw (pre-calibration):  |MBE|={mbe_raw:.2f}  RMSE={rmse_raw:.2f}  "
             f"score={score_raw:.2f}")

    # ── Per-station calibration ──────────────────────────────────────────────
    log.info("Fitting per-station linear calibration on daytime OOF ...")
    calib = fit_station_calibration(train, oof_pred)
    n_used  = sum(1 for v in calib.values() if v[3])
    n_total = len(calib)
    log.info(f"  calibrated stations: {n_used}/{n_total} (rest fall back to identity)")
    log.info("  per-station (a, b, n_daytime_rows, used):")
    for s in sorted(calib.keys()):
        a, b, n, used = calib[s]
        flag = "calib" if used else "ident"
        log.info(f"    {s}  a={a:+7.2f}  b={b:.3f}  n={n:6d}  [{flag}]")

    oof_calibrated = apply_station_calibration(train, oof_pred, calib)
    oof_calibrated = np.clip(oof_calibrated, 0, 1361)
    mbe_c, rmse_c, score_c = score_components(y_train, oof_calibrated)
    log.info(f"OOF after calibration:      |MBE|={mbe_c:.2f}  RMSE={rmse_c:.2f}  "
             f"score={score_c:.2f}  (delta={score_c-score_raw:+.2f})")

    # ── Refit on full training data ──────────────────────────────────────────
    best_iter = int(np.mean([fs["best_iter"] for fs in fold_scores]))
    log.info(f"Refitting on full train for {best_iter} rounds ...")
    full_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES)
    final_model = lgb.train(params, full_ds, num_boost_round=best_iter)

    log.info("Predicting test ...")
    test_pred = final_model.predict(X_test)
    test_pred = apply_station_calibration(test, test_pred, calib)

    # ── Nighttime override ───────────────────────────────────────────────────
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

    sub = pd.DataFrame({
        ID: test[ID],
        "TargetMBE":  test_pred,
        "TargetRMSE": test_pred,
    })
    sample = pd.read_csv(SAMP)
    sub = sample[[ID]].merge(sub, on=ID, how="left")
    if sub[["TargetMBE", "TargetRMSE"]].isna().any().any():
        log.error("  predictions missing — check ID alignment")

    sub_path = SUBS / f"{RUN_TAG}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}  ({len(sub):,} rows)")

    log_path = SUBS / f"{RUN_TAG}_log.txt"
    info = {
        "tag":                  RUN_TAG,
        "calibration":          "per-station LinearRegression(a, b) on daytime OOF",
        "min_calib_rows":       MIN_CALIB_ROWS,
        "b_range":              [B_MIN, B_MAX],
        "a_max_abs":            A_MAX_ABS,
        "day_elev_threshold":   DAY_ELEV_THRESHOLD,
        "wall_time_s":          round(time.time() - t0, 1),
        "n_train":              int(len(train)),
        "n_test":               int(len(test)),
        "features":             FEATURES,
        "lgbm_params":          params,
        "cv_folds":             fold_scores,
        "cv_oof_raw":           {"mbe": round(mbe_raw, 3),  "rmse": round(rmse_raw, 3),  "score": round(score_raw, 3)},
        "cv_oof_calibrated":    {"mbe": round(mbe_c, 3),    "rmse": round(rmse_c, 3),    "score": round(score_c, 3)},
        "score_delta":          round(score_c - score_raw, 3),
        "n_stations_calibrated": n_used,
        "n_stations_total":     n_total,
        "calibration_coeffs":   {s: {"a": round(a, 3), "b": round(b, 4),
                                      "n": int(n), "used": used}
                                  for s, (a, b, n, used) in calib.items()},
        "final_n_rounds":       best_iter,
        "submission":           str(sub_path),
    }
    log_path.write_text(json.dumps(info, indent=2, default=str))
    log.info(f"Saved run log:   {log_path}")
    log.info(f"Wall time: {info['wall_time_s']}s")


if __name__ == "__main__":
    main()
