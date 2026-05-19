# TAHMO Incoming Solar Radiation Prediction Challenge — Solution Notes

End-to-end pipeline for the Zindi TAHMO challenge: predict 15-min radiation
for the missing even-months of year 1 across 40 African weather stations.

## Repo map

```
/workspace
├── data/
│   ├── raw/            # Zindi-provided CSVs (gitignored)
│   ├── processed/      # Train_Test_Merged, Train/Test_enhanced (gitignored)
│   └── satellite/      # Per-source extracted CSVs (gitignored)
├── scripts/            # All Python extractors and training scripts
├── shell/              # All `run_*.sh` wrappers — entry points
├── submissions/        # Every submission and OOF artifact
├── notebooks/          # EDA / visualization
├── DATA_SOURCES.md     # What each external dataset gives us
├── MODELS.md           # Every training script's role + result
├── RESULTS.md          # Leaderboard progression
├── LESSONS.md          # What worked, what didn't
└── README.md           # ← you are here
```

## How to reproduce (fresh machine)

```bash
# 1. install
bash shell/install_all.sh
chmod 600 shell/_env.sh             # secrets file

# 2. fill credentials in shell/_env.sh (see header comments there)

# 3. build the canonical merged base
bash shell/run_prepare.sh

# 4. extract data (each in its own terminal — see shell/README.md)
bash shell/run_solar.sh
bash shell/run_pvlib.sh
bash shell/run_open_meteo.sh
bash shell/run_nasa_power.sh
bash shell/run_era5.sh             # ~3h
bash shell/run_cams.sh             # ~2h (ADS auth needed)
bash shell/run_cams_radiation_ts.sh # ~40min (ADS — the headline dataset)
bash shell/run_merra2.sh           # ~3-5h (Earthdata + GES DISC subscribe)
bash shell/run_lsa_saf.sh          # ~5h (EUMETSAT)
bash shell/run_lsa_saf_extra.sh    # ~5-10h (IPMA registration needed)
bash shell/run_tropomi.sh          # ~24h (CDSE)

# 5. extracted derived features
bash shell/run_temporal_neighbors.sh
bash shell/run_forward_weather.sh
bash shell/run_same_day_aggregates.sh
bash shell/run_energy_balance.sh

# 6. merge everything (safe to re-run any time)
bash shell/run_merge.sh

# 7. train + submit
bash shell/run_train_v10_csr.sh    # best single model
# combine: python scripts/ensemble.py file1.csv file2.csv --out blend.csv
```

## Quick stats

- 13 distinct external data sources extracted (8 with credentials)
- 12 modelling iterations submitted (v1 → v12)
- Best single LB: `lgbm_v10_csr.csv` at **composite 33.03** (RMSE 63.17, \|MBE\| 2.89)
- Leader at composite 29.93 — gap 3.1 W/m²

## Where to read more

- `DATA_SOURCES.md` — what each external dataset contributes (with API/auth notes)
- `MODELS.md` — every training script: what it added, what it scored
- `RESULTS.md` — full leaderboard progression
- `LESSONS.md` — what worked, what backfired, key gotchas
- `shell/README.md` — operational runbook
