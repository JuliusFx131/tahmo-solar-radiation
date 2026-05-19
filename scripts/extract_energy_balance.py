"""
Energy-balance inversion features.

The surface energy balance:
    Rnet = G + H + LE
  Rnet = net radiation (related to GHI by albedo + longwave correction)
  G    = ground heat flux
  H    = sensible heat flux  ~  rho * cp * dT/dt * h_bl
  LE   = latent heat flux    ~  rho * Lv * dq/dt * h_bl

Inverted, the temperature TENDENCY (dT/dt) times boundary-layer height
is a noisy but PHYSICAL estimator of energy absorbed by the air. Add
that as a feature and let LGBM learn the calibration.

Concrete features (ext_eb_* — energy-balance):
  ext_eb_dT_x_blh_1h      = (T_t+1h - T_t)  ×  ext_era5_blh   (J-ish proxy)
  ext_eb_dT_x_blh_3h      = (T_t+3h - T_t)  ×  ext_era5_blh
  ext_eb_dRH_x_blh_1h     = (RH_t+1h - RH_t) × ext_era5_blh
  ext_eb_warmup_rate_3h   = average dT/dt over morning hours (sunrise → +3h)
  ext_eb_cooldown_rate_3h = average dT/dt over evening (sunset-3h → sunset)
  ext_eb_temp_above_dewpoint = T_now - dewpoint_now (dryness proxy)
  ext_eb_specific_humidity_proxy = RH × saturation_vapor_pressure(T)
  ext_eb_dT_normalised_by_clearsky = dT/dt × 1/clearsky  (efficiency of heating)

Sources used:
  • temperature, RH, ext_era5_blh from Train/Test enhanced
  • ext_fw_temp_lead_*h, ext_fw_rh_lead_*h from forward_weather.csv
  • ext_om_dewpoint_2m from Open-Meteo merge
  • ext_sol_clearsky for normalisation
  • ext_sol_hour_angle / elevation for sunrise/sunset proxies

Run:
  bash /workspace/shell/run_energy_balance.sh

Output:
  data/satellite/energy_balance.csv
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parent.parent
PROC      = ROOT / "data" / "processed"
SAT_DIR   = ROOT / "data" / "satellite"
SAT_DIR.mkdir(parents=True, exist_ok=True)
ENH_TRAIN = PROC / "Train_enhanced.csv"
ENH_TEST  = PROC / "Test_enhanced.csv"
OUT       = SAT_DIR / "energy_balance.csv"


def sat_vapor_pressure_kpa(t_celsius):
    """Tetens formula. Saturation vapor pressure (kPa)."""
    return 0.6108 * np.exp(17.27 * t_celsius / (t_celsius + 237.3))


def main():
    t0 = time.time()
    log.info("Loading enhanced files (only the cols we need) ...")
    cols = [
        "station", "timestamp",
        "temperature (degrees Celsius)", "relativehumidity (-)",
        "ext_era5_blh", "ext_sol_clearsky", "ext_sol_elevation",
        "ext_fw_temp_lead_1h", "ext_fw_temp_lead_3h",
        "ext_fw_rh_lead_1h",   "ext_fw_temp_diff_1h", "ext_fw_temp_diff_3h",
        "ext_fw_rh_diff_1h",
        "ext_om_dewpoint_2m",
    ]
    train = pd.read_csv(ENH_TRAIN, usecols=cols, parse_dates=["timestamp"])
    test  = pd.read_csv(ENH_TEST,  usecols=cols, parse_dates=["timestamp"])
    full = pd.concat([train, test], ignore_index=True)
    full = full.sort_values(["station", "timestamp"]).reset_index(drop=True)
    log.info(f"  full rows: {len(full):,}")

    T   = full["temperature (degrees Celsius)"].values.astype(np.float32)
    RH  = full["relativehumidity (-)"].values.astype(np.float32)
    blh = full["ext_era5_blh"].values.astype(np.float32)
    cs  = full["ext_sol_clearsky"].values.astype(np.float32)
    dT1 = full["ext_fw_temp_diff_1h"].values.astype(np.float32)
    dT3 = full["ext_fw_temp_diff_3h"].values.astype(np.float32)
    dRH1= full["ext_fw_rh_diff_1h"].values.astype(np.float32)
    dew = full["ext_om_dewpoint_2m"].values.astype(np.float32)

    # 1. dT × BLH  — temperature gain × column depth ≈ energy absorbed
    full["ext_eb_dT_x_blh_1h"] = (dT1 * blh).astype(np.float32)
    full["ext_eb_dT_x_blh_3h"] = (dT3 * blh).astype(np.float32)
    full["ext_eb_dRH_x_blh_1h"] = (dRH1 * blh).astype(np.float32)

    # 2. T above dew point  — drier air heats faster under same radiation
    full["ext_eb_T_minus_dewpoint"] = (T - dew).astype(np.float32)

    # 3. Saturation-vapor humidity proxy
    es = sat_vapor_pressure_kpa(T)
    full["ext_eb_water_content_proxy"] = (RH * es).astype(np.float32)

    # 4. dT normalised by clearsky — efficiency of heating per W/m² available
    # Avoid divide-by-zero: clearsky=0 → no normalisation (set NaN, LGBM handles)
    eff = np.where(cs > 50, dT1 / cs, np.nan)
    full["ext_eb_dT_per_W_per_m2"] = eff.astype(np.float32)

    # 5. Morning warm-up and afternoon cool-down rates — per (station, date),
    #    avg dT/dt during sunrise→+3h and sunset-3h→sunset. We approximate
    #    sunrise as the first row of day with ext_sol_elevation crossing 0 going up.
    log.info("Computing morning warm-up / afternoon cool-down rates ...")
    full["date"] = full["timestamp"].dt.date
    full["elev"] = full["ext_sol_elevation"].values

    # Per station-date, locate sunrise and sunset row positions
    full["row_in_day"] = full.groupby(["station", "date"]).cumcount()
    full["is_day"] = (full["elev"] > 0).astype(np.int8)
    # Sunrise = first row where elev>0 within this day; Sunset = last such row.
    day_grp = full.groupby(["station", "date"])
    sunrise_idx = day_grp["is_day"].apply(lambda s: s.idxmax() if s.any() else np.nan)
    sunset_idx  = day_grp["is_day"].apply(lambda s: s[::-1].idxmax() if s.any() else np.nan)

    # 3 hours = 12 rows
    # Morning rate: (T at sunrise+12) - (T at sunrise) divided by 3h
    morning_rate = pd.Series(np.nan, index=full.index, dtype=np.float32)
    cool_rate    = pd.Series(np.nan, index=full.index, dtype=np.float32)
    T_ser = pd.Series(T, index=full.index)
    for (sta, d), grp in full.groupby(["station", "date"], sort=False):
        if not grp["is_day"].any():
            continue
        sun_up   = grp[grp["is_day"] == 1]
        sr_idx = sun_up.index[0]
        ss_idx = sun_up.index[-1]
        # morning (sunrise → +3h)
        post_sr = full.index[(full.index >= sr_idx) & (full.index <= min(sr_idx + 12, ss_idx))]
        if len(post_sr) >= 2:
            rate = (T_ser.loc[post_sr[-1]] - T_ser.loc[post_sr[0]]) / max(1, len(post_sr) - 1)
            morning_rate.loc[grp.index] = rate
        # afternoon (sunset-3h → sunset)
        pre_ss = full.index[(full.index >= max(ss_idx - 12, sr_idx)) & (full.index <= ss_idx)]
        if len(pre_ss) >= 2:
            rate = (T_ser.loc[pre_ss[-1]] - T_ser.loc[pre_ss[0]]) / max(1, len(pre_ss) - 1)
            cool_rate.loc[grp.index] = rate

    full["ext_eb_morning_warming"] = morning_rate.astype(np.float32)
    full["ext_eb_afternoon_cooling"] = cool_rate.astype(np.float32)

    eb_cols = [c for c in full.columns if c.startswith("ext_eb_")]
    out = full[["station", "timestamp"] + eb_cols].copy()
    out.to_csv(OUT, index=False)
    log.info(f"Saved: {OUT}  ({len(out):,} rows × {len(out.columns)} cols)")
    for c in eb_cols:
        cov = out[c].notna().mean() * 100
        log.info(f"  {c}: {cov:.1f}% non-null")
    log.info(f"Wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
