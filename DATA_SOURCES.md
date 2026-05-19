# Data Sources

Every external dataset we extracted, what it provides, its cost in time/auth,
and what feature columns it produces in `Train_enhanced.csv` / `Test_enhanced.csv`.

## Native data (no extraction)

| Source | Cadence | What |
| --- | --- | --- |
| `data/raw/Train.csv` | 15-min | temperature, RH, precipitation, radiation, station metadata. 40 stations × ~330 days × odd months only |
| `data/raw/Test.csv` | 15-min | same columns minus radiation. 40 stations × even months |

## Computed / no API

| Script | Output prefix | What | Wall time |
| --- | --- | --- | --- |
| `scripts/data_pipeline.py::compute_solar_features` | `ext_sol_*` | NOAA solar geometry: zenith, azimuth, elevation, hour angle, declination, clear-sky GHI, day length | <30s |
| `scripts/extract_pvlib.py` | `ext_pv_*` | pvlib-based clear-sky: apparent zenith/elevation, Linke turbidity, Ineichen-Perez + Haurwitz GHI/DNI/DHI, airmass | ~20s |
| `scripts/extract_temporal_neighbors.py` | `ext_tn_*` | Per (station, hour, doy) rolling-window mean radiation from training data; ±7/15/30/60 day windows. **CAUTION**: leaky in CV unless recomputed per fold | ~15s |
| `scripts/extract_forward_weather.py` | `ext_fw_*` | FUTURE temperature/humidity/precipitation from the same test row's surrounding observations (test rows lead from other test rows in the same month). The change over the next 1h is a proxy for current radiation | ~10s |
| `scripts/extract_same_day_aggregates.py` | `ext_dd_*` | Per (station, date) daily max/min/mean/std of temperature, humidity, precip, plus daily max/sum of all radiation estimates (OM/NP/ERA5/SARAH) | ~20s |
| `scripts/extract_energy_balance.py` | `ext_eb_*` | Physical proxies: dT/dt × BLH, T-dewpoint, morning warming rate, afternoon cooling rate. "Inversion" via energy balance | ~45s |

## External API — no auth

| Source | API | Output prefix | What | Wall time |
| --- | --- | --- | --- | --- |
| Open-Meteo ERA5 archive | `archive-api.open-meteo.com` | `ext_om_*` | 15 variables hourly: shortwave/direct/diffuse GHI, layered cloud cover (low/mid/high), wind, CAPE (sometimes empty), temp, dewpoint, RH, pressure, precip | ~5min |
| NASA POWER | `power.larc.nasa.gov` | `ext_np_*` | MERRA-2 + GEOS satellite correction hourly: all-sky/clear-sky GHI, DNI, DHI, clearness index, cloud amount, AOD, bias-corrected precip | ~30min |

## External API — auth required

| Source | Auth | Output prefix | What | Wall time |
| --- | --- | --- | --- | --- |
| **CAMS Solar Radiation Timeseries** (the killer feature) | ADS_KEY + dataset licence accept | `ext_csr_*` | **15-min cadence** pre-computed GHI/BHI/DHI/BNI at exact station coordinates, all-sky AND clear-sky + reliability flag. Single largest feature gain we've ever seen (10× the next feature) | ~40min |
| ERA5 reanalysis (CDS) | CDS_KEY | `ext_era5_*` | Hourly: ssrd (shortwave radiation downwards), tcc (cloud cover), tcwv (water vapour), blh (boundary layer height), sp (surface pressure) | ~3h |
| CAMS EAC4 aerosols (ADS) | ADS_KEY | `ext_cams_*` | 3-hourly speciated AODs (total, dust, BC, OC, sulphate, sea-salt) + total column water vapour | ~2h |
| SARAH-3 daily (EUMETSAT) | EUMETSAT_KEY/SECRET | `ext_lsa_*` (sis/sid/dni) | Daily mean shortwave (SISdm), direct (SIDdm), DNI (DNIdm) over Africa | ~5h |
| MERRA-2 aerosols (NASA GES DISC) | EARTHDATA_TOKEN + GESDISC subscribe | `ext_m2_*` | Hourly speciated AODs (total/dust/OC/BC/SO4/SS) + Ångström exponent + PM2.5 (dust, OC) | ~3-5h |
| TROPOMI cloud + aerosol (CDSE) | CDSE_USER/PASSWORD | `ext_tro_*` | Daily satellite cloud fraction, cloud top pressure, absorbing aerosol index. Extraction has bugs — only `aerosol_index_354_388` populates reliably (38% coverage). Other columns are empty after QA filter | ~24h subsampled |
| **LSA-SAF MDSSFTD** (IPMA) | LSASAF_USER/PASS at mokey.lsasvcs.ipma.pt | `ext_mdssftd_*` | **15-min cadence** MSG satellite-derived DSSF (total downwelling shortwave, W/m²) + **FRACTION_DIFFUSE** (the leader's "Inversion" signature feature). Same physical quantity as the competition target, derived from MSG SEVIRI on the same MSG grid the test set was sampled from. Smoke test: TA00349 at 12:00 UTC 2018-01-15 → 936.6 W/m² DSSF, 22% diffuse, 730 W/m² direct beam. Wall time ~14.7s/day → ~10.5h for full 2018-2024 | ~10.5h |

## External API — pending (not yet extracted)

| Source | Auth | Why valuable |
| --- | --- | --- |
| LSA-SAF MLST (IPMA) | LSASAF_USER/PASS | Land surface temperature. Marginal expected gain (~0.3-1 W/m²) |
| LSA-SAF MDSLF (IPMA) | LSASAF_USER/PASS | Downwelling longwave. Mostly redundant with what ERA5 gives us |

## Coverage / quality notes

- **CAMS aerosols are 3-hourly** — every training script forward-fills CAMS columns within each station to bring effective coverage from 33% → 100%.
- **ext_om_cape is 0%** — Open-Meteo's archive doesn't expose CAPE for these locations/dates. We drop it from the model.
- **ext_pv_airmass_* is 50% NaN at night** (geometric — undefined when zenith > 90°). Dropped from feature list.
- **`ext_csr_clearness_kt` is 49%** (NaN when clear-sky GHI < 5 W/m², i.e., night). Derived feature; primary `ext_csr_ghi` is 100%.
- **MERRA-2 `ext_m2_aod_so4` is 0%** — extraction bug (likely a variable-name mismatch in the script). Dropped from features.

## Auth credentials registry (in `_env.sh`)

```
EUMETSAT_KEY / EUMETSAT_SECRET   — SARAH-3 (and would-be MDSSFTD-via-EUMETSAT)
CDS_KEY                          — ERA5
ADS_KEY (= CDS_KEY typically)    — CAMS aerosols + CAMS Solar Radiation Timeseries
CDSE_USER / CDSE_PASSWORD        — TROPOMI
EARTHDATA_TOKEN                  — MERRA-2 (needs GES DISC subscription on top)
LSASAF_USER / LSASAF_PASS        — LSA-SAF MDSSFTD/MLST/MDSLF at mokey.lsasvcs.ipma.pt
                                   (separate auth from landsaf.ipma.pt; via Basic auth
                                   to datalsasaf.lsasvcs.ipma.pt HTTPS host)
```

Each external service needed a portal account and (for some) one-time licence
acceptance. Keys are in `keys.md` and `_env.sh` (both gitignored).
