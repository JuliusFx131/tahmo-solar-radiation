# /workspace/shell — all bash entry points

Everything you can run from a terminal lives here. One file = one job, so you can
open multiple terminals and fire the data sources in parallel — but **mind the
per-API rate limits** noted below.

```
shell/
├── _env.sh                       # sourced by every run_*.sh — credentials + $PY
├── install_base.sh               # core deps (pandas, numpy, sklearn, lightgbm, …)
├── install_pipeline.sh           # satellite-API deps (eumdac, cdsapi, pvlib, …)
├── install_all.sh                # base + pipeline in one go
│
├── run_prepare.sh                # build data/processed/Train_Test_Merged.csv
│
│  # ── External data extractions ────────────────────────────────────────────
├── run_solar.sh                  # computed solar geometry (no creds, ~30s)
├── run_pvlib.sh                  # pvlib clear-sky + airmass (no creds, ~30s)
├── run_open_meteo.sh             # Open-Meteo ERA5 archive (no creds, ~5 min, free API)
├── run_nasa_power.sh             # NASA POWER (MERRA-2+GEOS) hourly (no creds, ~30 min)
├── run_lsa_saf.sh                # SARAH-3 daily SW radiation (EUMETSAT, ~5 h)
├── run_lsa_saf_extra.sh          # MDSSFTD + MLST + MDSLF (EUMETSAT, ~5-10 h)
├── run_tropomi.sh                # Copernicus TROPOMI cloud + AI (CDSE, ~24 h subsampled)
├── run_era5.sh                   # CDS ERA5 reanalysis (CDS_KEY, ~3 h)
├── run_cams.sh                   # ADS CAMS aerosols (ADS_KEY, ~2 h)
├── run_merra2.sh                 # NASA MERRA-2 aerosols (EARTHDATA_TOKEN, ~3-5 h)
├── run_modis.sh                  # NASA MODIS daily (EARTHDATA_TOKEN, ~6 h, needs pyhdf)
│
│  # ── Pipeline glue ────────────────────────────────────────────────────────
├── run_merge.sh                  # stitch all available CSVs → Train/Test_enhanced
│
│  # ── Model training ───────────────────────────────────────────────────────
├── run_train_lgbm.sh             # baseline LGBM
├── run_train_lgbm_kt.sh          # KT-target variant
├── run_train_lgbm_station_calib.sh
├── run_train_lgbm_feat_eng.sh    # v1: solar-clock + lags + rolling
├── run_train_lgbm_feat_eng_v2.sh # v2: + anomaly_score + tcwv/blh rolls + per-hour night
├── run_train_lgbm_feat_eng_v3.sh # v3: + pvlib + Open-Meteo features
├── run_train_lgbm_mae.sh         # MAE objective companion
└── run_train_lgbm_climatology.sh # + station-hour climatology feature
```

## First time on a fresh machine

```bash
bash shell/install_all.sh
chmod 600 shell/_env.sh           # _env.sh contains secrets
```

## End-to-end runbook (in order)

```bash
# 1. Build the canonical merged base file (cheap)
bash shell/run_prepare.sh

# 2. Free / fast extractors (no auth, run in parallel)
bash shell/run_solar.sh           # ~30 s
bash shell/run_pvlib.sh           # ~30 s
bash shell/run_open_meteo.sh      # ~5 min (free API, rate-limited; do NOT run twice in parallel)
bash shell/run_nasa_power.sh      # ~30 min (free API, polite ~4s between calls)

# 3. Authenticated extractors (run each in its own terminal; safe in parallel)
bash shell/run_lsa_saf.sh                  # ~5 h
bash shell/run_lsa_saf_extra.sh            # ~5-10 h for all three (--product to limit)
bash shell/run_era5.sh                     # ~3 h
bash shell/run_cams.sh                     # ~2 h
bash shell/run_merra2.sh                   # ~3-5 h (needs NASA GESDISC subscription)
bash shell/run_tropomi.sh                  # ~24 h subsampled (per-day checkpoint)
bash shell/run_modis.sh                    # only if pyhdf installed

# 4. Merge once extractors finish (safe to re-run any time — skips missing CSVs)
bash shell/run_merge.sh

# 5. Train + submit
bash shell/run_train_lgbm_feat_eng_v3.sh   # current best single model
# then ensemble (see "Ensembling" below)
```

## Per-extractor notes

### Free / no auth

| Script | API | Rate-limit hint | Notes |
| --- | --- | --- | --- |
| `run_solar.sh` | none (local math) | n/a | Re-run any time, regenerates `solar_features.csv` |
| `run_pvlib.sh` | none (local math) | n/a | Linke turbidity is bundled with pvlib |
| `run_open_meteo.sh` | `archive-api.open-meteo.com` | **8 s between stations enforced**; do NOT run two copies in parallel — triggers HTTP 429 | Resumable per-station |
| `run_nasa_power.sh` | `power.larc.nasa.gov` | 4 s between stations | Resumable per-station |

### EUMETSAT (`EUMETSAT_KEY`, `EUMETSAT_SECRET`)

| Script | Resumable | Notes |
| --- | --- | --- |
| `run_lsa_saf.sh` | per-month checkpoint | SARAH-3 (`SISdm`, `SIDdm`, `DNIdm`) |
| `run_lsa_saf_extra.sh` | per-product per-month checkpoint | MDSSFTD (diffuse fraction), MLST (LST), MDSLF (longwave). Default: all three sequentially. Pass `--product mdssftd` to run one |

### CDS (`CDS_KEY`)

| Script | Resumable | Notes |
| --- | --- | --- |
| `run_era5.sh` | per-month `.nc` cache | Re-extract from cache is fast if interrupted |

### ADS (`ADS_KEY`, separate from CDS — **needs CAMS license accepted once**)

| Script | Resumable | Notes |
| --- | --- | --- |
| `run_cams.sh` | per-month `.nc` cache | Now pulls 7 species (total/dust/BC/OC/sulfate/sea-salt + TCWV). **If you previously cached 3 species only, delete `data/satellite/cams_*.nc` so the 4 new species get fetched** |

### CDSE (`CDSE_USER`, `CDSE_PASSWORD`)

| Script | Resumable | Notes |
| --- | --- | --- |
| `run_tropomi.sh` | per-day checkpoint | Default 3 granules/day. Override with `TROPOMI_MAX_GRANULES_PER_DAY=5 bash …` |

### NASA Earthdata (`EARTHDATA_TOKEN`)

| Script | Resumable | Notes |
| --- | --- | --- |
| `run_merra2.sh` | per-day CSV cache | **One-time:** subscribe to "NASA GESDISC DATA ARCHIVE" application at <https://urs.earthdata.nasa.gov/> or downloads return 401 |
| `run_modis.sh` | per-month checkpoint | Needs `python-hdf4` (may fail on numpy 2.x — see install log) |

## Merge step

`run_merge.sh` always picks up whatever CSVs exist in `data/satellite/`. Missing
sources are logged and skipped — you can run it before extractors finish to get
a partial enhanced dataset.

## Ensembling

```bash
# Equal-weight blend of three LGBMs
python scripts/ensemble.py \
  submissions/lgbm_feat_eng_v3.csv \
  submissions/lgbm_mae_v1.csv \
  submissions/lgbm_climatology_v1.csv \
  --out submissions/lgbm_ensemble_v4_v3swap.csv

# Weighted
python scripts/ensemble.py file1.csv file2.csv file3.csv \
  --weights 0.5 0.3 0.2 \
  --out submissions/my_blend.csv
```

## Backgrounding instead of using terminals

```bash
mkdir -p logs
nohup bash shell/run_era5.sh        > logs/era5.log        2>&1 &
nohup bash shell/run_lsa_saf.sh     > logs/lsa_saf.log     2>&1 &
nohup bash shell/run_lsa_saf_extra.sh > logs/lsa_saf_extra.log 2>&1 &
nohup bash shell/run_tropomi.sh     > logs/tropomi.log     2>&1 &
nohup bash shell/run_merra2.sh      > logs/merra2.log      2>&1 &
```

## Override the interpreter

Every script honours `$PY`:

```bash
PY=python3.11 bash shell/run_era5.sh
```

## Visualization-only flow

The visualization notebook reads `data/processed/Train_Test_Merged.csv`. To
(re)generate it without running any extractor:

```bash
bash shell/run_prepare.sh
```
