"""
Prepare a shared feature matrix for the v20 multi-model run.
Same FE pipeline as v18 (MDSSFTD merge + derived + rolling + time + CAMS ffill)
but WITHOUT pseudo-labels — we want a clean baseline for cross-model comparison.

Saves train/test as parquet for fast reload from each of the 4 trainer scripts:
  data/processed/v20_train.parquet
  data/processed/v20_test.parquet
  data/processed/v20_feature_list.json
"""
import gc, json, logging, time
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT  = Path(__file__).resolve().parent.parent
PROC  = ROOT / "data" / "processed"
MDSS  = ROOT / "data" / "satellite" / "mdssftd_per_station"
TRAIN = PROC / "Train_enhanced.csv"
TEST  = PROC / "Test_enhanced.csv"

TARGET = "radiation (W/m2)"
ID     = "ID"


def merge_mdssftd(df):
    files = sorted(p for p in MDSS.iterdir() if p.suffix == ".csv")
    parts = []
    for p in files:
        d = pd.read_csv(p, parse_dates=["timestamp"])
        d["station"] = p.stem
        for c in ("ext_mdssftd_dssf","ext_mdssftd_fdiff",
                  "ext_mdssftd_dssf_direct","ext_mdssftd_qflag"):
            if c in d.columns:
                d[c] = d[c].astype("float32")
        parts.append(d[["station","timestamp",
                        "ext_mdssftd_dssf","ext_mdssftd_fdiff",
                        "ext_mdssftd_dssf_direct","ext_mdssftd_qflag"]])
    mdss = pd.concat(parts, ignore_index=True, copy=False)
    return df.merge(mdss, on=["station","timestamp"], how="left", copy=False)


def add_mdssftd_derived(df):
    eps = np.float32(1.0)
    dssf       = df["ext_mdssftd_dssf"].astype("float32")
    dssf_direct= df["ext_mdssftd_dssf_direct"].astype("float32")
    fdiff      = df["ext_mdssftd_fdiff"].astype("float32")
    csr_ghi    = df["ext_csr_ghi"].astype("float32")
    csr_cs_ghi = df["ext_csr_clearsky_ghi"].astype("float32")
    csr_bhi    = df["ext_csr_bhi"].astype("float32")
    df["ext_mdssftd_kt"]                   = (dssf / np.maximum(csr_cs_ghi, eps)).astype("float32")
    df["ext_mdssftd_anomaly"]              = (dssf - csr_ghi).astype("float32")
    df["ext_mdssftd_clearsky_anomaly"]     = (dssf - csr_cs_ghi).astype("float32")
    df["ext_mdssftd_fdiff_x_csr_ghi"]      = (fdiff * csr_ghi).astype("float32")
    df["ext_mdssftd_direct_minus_csr_bhi"] = (dssf_direct - csr_bhi).astype("float32")
    return df


def add_time(df):
    ts = df["timestamp"]
    df["hour"]   = ts.dt.hour.astype("int8")
    df["minute"] = ts.dt.minute.astype("int8")
    df["month"]  = ts.dt.month.astype("int8")
    df["doy"]    = ts.dt.dayofyear.astype("int16")
    df["year"]   = ts.dt.year.astype("int16")
    df["dow"]    = ts.dt.dayofweek.astype("int8")
    df["dom"]    = ts.dt.day.astype("int8")
    df["hour_sin"] = np.sin(2*np.pi*(df["hour"]+df["minute"]/60)/24).astype("float32")
    df["hour_cos"] = np.cos(2*np.pi*(df["hour"]+df["minute"]/60)/24).astype("float32")
    df["doy_sin"]  = np.sin(2*np.pi*df["doy"]/365.25).astype("float32")
    df["doy_cos"]  = np.cos(2*np.pi*df["doy"]/365.25).astype("float32")
    utc_hour = ts.dt.hour + ts.dt.minute/60
    df["solar_clock"]            = ((utc_hour + df["longitude"]/15.0) % 24).astype("float32")
    df["hours_from_solar_noon"]  = (df["solar_clock"] - 12).astype("float32")
    first = df.groupby("station")["timestamp"].transform("min")
    df["days_since_install"] = ((df["timestamp"]-first).dt.total_seconds()/86400).astype("float32")
    return df


def ffill_cams(df):
    cams = [c for c in df.columns if c.startswith("ext_cams_")]
    df = df.sort_values(["station","timestamp"]).reset_index(drop=True)
    df[cams] = df.groupby("station")[cams].transform(lambda s: s.ffill().bfill())
    return df


def add_rolling(df):
    df = df.sort_values(["station","timestamp"]).reset_index(drop=True)
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

    for col, short in roll_cols.items():
        if col not in df.columns: continue
        base = df[col].values.astype(np.float32, copy=False)
        for w_name, w in windows.items():
            out = np.empty_like(base, dtype=np.float32)
            for sta, idx in g_idx.items():
                out[idx] = pd.Series(base[idx]).rolling(w, min_periods=1).mean().to_numpy(dtype=np.float32)
            df[f"{short}_mean_{w_name}"] = out
            del out
        lag1 = np.empty_like(base, dtype=np.float32)
        lag3 = np.empty_like(base, dtype=np.float32)
        for sta, idx in g_idx.items():
            lag1[idx] = pd.Series(base[idx]).shift(4).to_numpy(dtype=np.float32)
            lag3[idx] = pd.Series(base[idx]).shift(12).to_numpy(dtype=np.float32)
        df[f"{short}_lag_1h"]  = lag1
        df[f"{short}_lag_3h"]  = lag3
        df[f"{short}_diff_1h"] = (base - lag1).astype(np.float32)
        del base, lag1, lag3

    base_precip = df["precipitation (mm)"].values.astype(np.float32, copy=False)
    for label, w in [("precip_sum_6h", 24), ("precip_sum_24h", 96)]:
        out = np.empty_like(base_precip, dtype=np.float32)
        for sta, idx in g_idx.items():
            out[idx] = pd.Series(base_precip[idx]).rolling(w, min_periods=1).sum().to_numpy(dtype=np.float32)
        df[label] = out
        del out

    for col, short in [("ext_mdssftd_dssf","mdss_dssf"),("ext_mdssftd_kt","mdss_kt")]:
        base = df[col].values.astype(np.float32, copy=False)
        for w_name, w in [("1h", 4),("3h", 12)]:
            out = np.empty_like(base, dtype=np.float32)
            for sta, idx in g_idx.items():
                out[idx] = pd.Series(base[idx]).rolling(w, min_periods=1).mean().to_numpy(dtype=np.float32)
            df[f"{short}_mean_{w_name}"] = out
            del out
        del base
    return df


# Feature manifest — same as v18 minus the m2_* (proved net-negative)
BASE = [
    "precipitation (mm)","relativehumidity (-)","temperature (degrees Celsius)",
    "installation_height","elevation","latitude","longitude",
    "ext_sol_elevation","ext_sol_clearsky","ext_lsa_dni","ext_lsa_sid",
    "ext_era5_ssrd","ext_era5_tcc","ext_era5_tcwv","ext_era5_blh","ext_era5_sp",
    "ext_cams_aod550","ext_cams_duaod550","ext_cams_bcaod550",
    "ext_cams_omaod550","ext_cams_suaod550","ext_cams_ssaod550","ext_cams_tcwv",
    "ext_pv_apparent_elevation","ext_pv_etr","ext_pv_linke_turbidity",
    "ext_pv_clearsky_ghi","ext_pv_clearsky_dni","ext_pv_clearsky_dhi","ext_pv_clearsky_ghi_haur",
    "ext_om_ghi","ext_om_direct_horiz","ext_om_dni","ext_om_dhi",
    "ext_om_cc_total","ext_om_cc_low","ext_om_cc_mid","ext_om_cc_high",
    "ext_om_wind_speed_10m","ext_om_wind_dir_10m",
    "ext_om_temperature_2m","ext_om_dewpoint_2m","ext_om_humidity_2m",
    "ext_om_pressure_surface","ext_om_precip",
    "ext_np_allsky_ghi","ext_np_allsky_dhi","ext_np_allsky_dni",
    "ext_np_clrsky_ghi","ext_np_cloud_amount","ext_np_aod_550","ext_np_precip_corr",
    "ext_fw_temp_lead_15m","ext_fw_temp_lead_30m","ext_fw_temp_lead_1h","ext_fw_temp_lead_3h",
    "ext_fw_temp_diff_1h","ext_fw_temp_diff_3h",
    "ext_fw_rh_lead_15m","ext_fw_rh_lead_30m","ext_fw_rh_lead_1h","ext_fw_rh_lead_3h","ext_fw_rh_diff_1h",
    "ext_fw_precip_lead_15m","ext_fw_precip_lead_30m","ext_fw_precip_lead_1h","ext_fw_precip_lead_3h",
    "ext_fw_precip_sum_lead_3h","ext_fw_precip_sum_lead_24h",
    "ext_dd_temp_max","ext_dd_temp_min","ext_dd_temp_amp","ext_dd_temp_mean","ext_dd_temp_std",
    "ext_dd_rh_max","ext_dd_rh_min","ext_dd_rh_mean",
    "ext_dd_precip_sum","ext_dd_precip_max",
    "ext_dd_om_ghi_max","ext_dd_om_ghi_sum","ext_dd_om_cc_mean",
    "ext_dd_np_ghi_max","ext_dd_np_ghi_sum","ext_dd_ssrd_max","ext_dd_ssrd_sum",
    "ext_eb_dT_x_blh_1h","ext_eb_dT_x_blh_3h","ext_eb_dRH_x_blh_1h",
    "ext_eb_T_minus_dewpoint","ext_eb_water_content_proxy","ext_eb_dT_per_W_per_m2",
    "ext_eb_morning_warming","ext_eb_afternoon_cooling",
    "ext_csr_ghi","ext_csr_bhi","ext_csr_dhi","ext_csr_bni",
    "ext_csr_clearsky_ghi","ext_csr_clearsky_bhi","ext_csr_clearsky_dhi","ext_csr_clearsky_bni",
    "ext_csr_reliability","ext_csr_clearness_kt","ext_csr_diffuse_fraction",
    "ext_mdssftd_dssf","ext_mdssftd_fdiff","ext_mdssftd_dssf_direct","ext_mdssftd_qflag",
    "ext_mdssftd_kt","ext_mdssftd_anomaly","ext_mdssftd_clearsky_anomaly",
    "ext_mdssftd_fdiff_x_csr_ghi","ext_mdssftd_direct_minus_csr_bhi",
]
TIME = ["hour","minute","month","doy","year","dow","dom",
        "hour_sin","hour_cos","doy_sin","doy_cos",
        "solar_clock","hours_from_solar_noon","days_since_install"]
ROLL_SUFFIXES = ["_mean_1h","_mean_3h","_mean_6h","_mean_24h","_lag_1h","_lag_3h",
                 "_diff_1h","_sum_6h","_sum_24h"]
CATEGORICAL = ["station","country"]


def main():
    t0 = time.time()
    log.info("Loading enhanced train + test ...")
    train = pd.read_csv(TRAIN, parse_dates=["timestamp"])
    test  = pd.read_csv(TEST,  parse_dates=["timestamp"])
    log.info(f"  train {train.shape}  test {test.shape}")

    for df in (train, test):
        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = df[c].astype("float32")

    for split, df in (("train", train), ("test", test)):
        log.info(f"  → {split}: merging MDSSFTD + derived + ffill_cams + time + rolling ...")
        df = merge_mdssftd(df)
        df = add_mdssftd_derived(df)
        df = ffill_cams(df)
        df = add_time(df)
        df = add_rolling(df)
        if split == "train":
            train = df
        else:
            test = df
        log.info(f"     done. shape={df.shape}")

    # Build the explicit feature list (deterministic, no FW/EB suffix collisions)
    rolls = [c for c in train.columns
             if any(c.endswith(s) for s in ROLL_SUFFIXES)
             and not c.startswith("ext_fw_") and not c.startswith("ext_eb_")]
    features = list(dict.fromkeys(BASE + TIME + rolls + CATEGORICAL))
    log.info(f"Features kept: {len(features)} total  ({len(rolls)} rolling/lag/diff)")

    # Make station/country categorical with a SHARED category mapping
    for col in CATEGORICAL:
        cats = sorted(set(train[col].astype(str)) | set(test[col].astype(str)))
        train[col] = pd.Categorical(train[col].astype(str), categories=cats)
        test[col]  = pd.Categorical(test[col].astype(str),  categories=cats)

    # Dedup ("ext_sol_elevation" is in BASE, so don't add it again to test)
    extras_train = [c for c in [TARGET, ID, "timestamp"] if c not in features]
    extras_test  = [c for c in [ID, "timestamp", "ext_sol_elevation"] if c not in features]
    train_out  = train[features + extras_train].copy()
    test_out   = test[features + extras_test].copy()

    train_path = PROC / "v20_train.parquet"
    test_path  = PROC / "v20_test.parquet"
    feats_path = PROC / "v20_feature_list.json"
    train_out.to_parquet(train_path, index=False)
    test_out.to_parquet(test_path, index=False)
    feats_path.write_text(json.dumps({
        "features": features,
        "categorical": CATEGORICAL,
        "target": TARGET,
        "id": ID,
        "n_train": int(len(train_out)),
        "n_test":  int(len(test_out)),
        "build_seconds": round(time.time()-t0, 1),
    }, indent=2))
    log.info(f"Saved {train_path}  ({len(train_out):,} rows)")
    log.info(f"Saved {test_path}   ({len(test_out):,} rows)")
    log.info(f"Saved {feats_path}")
    log.info(f"Wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
