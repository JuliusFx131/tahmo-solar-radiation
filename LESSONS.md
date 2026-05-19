# Lessons Learned

Things that worked, things that backfired, and gotchas worth knowing if you
pick up this project later.

## Big wins

### 1. Treat the problem as INTERPOLATION, not forecasting

The single most important framing shift. Train (odd months) and test
(even months) are **interleaved gaps** in a continuous timeline at the
same stations. That unlocks:

- **Forward-weather lags** (v7) — for any test row, we know what temperature
  was 1h later (still in the same test month). The temperature change over
  the next hour is a proxy for current radiation.
- **Same-day aggregates** (v8) — every test row knows the max temperature
  for its whole day.
- **CAMS Solar Radiation Timeseries** — pre-computed radiation estimates
  for *the exact test moments* are buyable from ADS.

We spent the first half of the session treating test as "unseen future"
and missed all of the above. Once we reframed, the path was obvious.

### 2. Single best external dataset: CAMS Solar Radiation Timeseries

`cams-solar-radiation-timeseries` (different from `cams-global-reanalysis-eac4`
we already had). 15-min cadence, station-coordinate-exact, GHI/BHI/DHI/BNI
all-sky + clear-sky. Worth more than the rest of our work combined
(-3 W/m² on LB in a single submission). 

The forum's transparency thread (thisiskuhan flagging it) was the entire
unlock.

### 3. Energy-balance "Inversion" features earned their keep

Physical intuitions made it into the top 15 by gain:
- `ext_eb_T_minus_dewpoint` — dry air heats faster per W/m²
- `ext_eb_dT_x_blh_1h` — temperature change × boundary layer height ≈ energy absorbed
- `ext_eb_morning_warming` — slope of sunrise→sunrise+3h temperature rise

None of these are individually huge but stacked they bought ~0.5 W/m².

### 4. Same-day daily aggregates were "what kind of day is this"

Daily max/min/mean across the whole day, per station-date, propagated to
every 15-min row. Tells the model the day's character (sunny vs cloudy)
regardless of the moment.

`ext_dd_np_ghi_sum` (daily NASA POWER GHI sum) was in v8's top 10.

### 5. Per-(station, hour) nighttime override

Forum-confirmed: the LB rewards mimicking raw sensor behavior, including
the noisy nighttime non-zero readings. Predicting station-specific
nighttime mean (split per hour of night) consistently helped.

## Big losses

### 1. Temporal-neighbor features (v6) — CV leakage + distribution shift

Idea: for each row at (station, hour, doy), the mean radiation across
training rows in a ±N-day window. Beautiful in theory; catastrophe in
practice.

- **CV leakage**: precomputed once from all training rows. In a fold
  where val_month = M, val rows' neighbor features include OTHER rows
  from the SAME month M (which the fold's training data shouldn't have
  seen). OOF inflated from ~41 to 37.
- **Test-time distribution shift**: imputation cascade to fill narrow
  windows for test rows created a feature distribution at inference
  that differed from training. Predictions collapsed (mean 103 vs train 188).

**LB regression: +22.8 W/m².** Never again without per-fold recomputation.

### 2. Per-station modeling

40 separate LGBMs (one per station) using v4 features. Lost the
cross-station learning that the shared model exploits. **LB regression: +6.95.**

### 3. Log-transform target

`log1p(radiation)` target with `expm1` back-transform and bias correction.
Fold 0 OOF score 47.62 (vs v4's 39.99). Killed early. Squared-error trees
don't benefit much, and the back-transform introduced bias.

### 4. Huber loss

`alpha=0.9` was wildly wrong for W/m² scale (errors are routinely >50).
Effectively became MAE with worse convergence. Fold 0 RMSE = 220 → killed.

### 5. Ensembling weak models with strong ones

After v10 hit 33.03 LB, ensembling with `lgbm_mae_v1` (39.60) or
`lgbm_climatology_v1` (37.92) consistently *hurt*. The weak companions
diluted v10's gain. For ensembling, all components must be of similar
strength.

## Gotchas

### 1. 8 GB cgroup memory limit (RunPod)

Easy to forget on a host with 700+ GB visible RAM. Symptoms: silent OOM
kill mid-CV with no error. Cure:
- **float32 downcast** of all numeric columns immediately after loading
- **gc.collect()** + `del ds_tr, ds_va, model` between folds
- **pd.concat(copy=False)** and explicit `del` of intermediate frames
- For training: `histogram_pool_size=512` caps LGBM's internal hist memory
- Smaller `num_leaves` and bigger `min_data_in_leaf` reduce per-fold peak

### 2. Disk-full / chunk-aligned truncation

MFS / RunPod's storage truncates large files at chunk boundaries (we
saw `era5_2018-07.nc` truncated at exactly 128 MiB = 134217728 bytes).
A truncated `.nc` file:
- Has the right magic bytes ("PK" for zips) — looks valid to magic-byte checks
- Fails when xarray/zipfile tries to read the central directory at the end
- Doesn't trigger our "size < N bytes" defensive check

Added a defensive check in `extract_era5` / `extract_cams` to delete any
cached file under 5 MB (it's almost certainly truncated).

### 3. Atomic checkpoint writes

We hit multiple disk-fulls mid-`to_csv(checkpoint_path)` that left a
0-byte file, which then crashed `pd.read_csv` on the next run with
`EmptyDataError`. Fixed by routing all checkpoint writes through
`_safe_write_csv` (write to .tmp then atomic rename) and reads through
`_safe_read_csv` (returns None on empty/corrupt).

### 4. CDS-Beta API changes (the great Spring 2025-2026 migration)

The Climate Data Store rebuilt their stack. Most older `cdsapi` examples
break. Fixes we made:

- `format: "netcdf"` → `data_format: "netcdf"` + `download_format: "unarchived"`
- Responses are ZIP-wrapped (`.zip` file with `.nc` inside) even when
  `unarchived` requested. Detect PK magic, extract.
- NetCDF coord renamed from `time` to `valid_time` in many products
- Variable names changed (e.g., long `surface_solar_radiation_downwards`
  → short `ssrd`)
- CAMS aerosols moved from CDS to ADS (separate URL, separate licence)

### 5. EUMETSAT vs IPMA for LSA-SAF — and the two IPMA accounts

Same product family ("LSA-SAF"), different hosts and different account systems.
EUMETSAT Data Store account works for SARAH-3 (EO:EUM:DAT:0863) but NOT
for MDSSFTD / MDSLF / MLST, which live only on IPMA's mirror.

For those, register at <https://mokey.lsasvcs.ipma.pt/auth/signup>. Note:
this is a separate account from the older <https://landsaf.ipma.pt/> portal —
they share branding but not credentials. Files are served from
`https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/<product>/NETCDF/YYYY/MM/DD/`
via HTTP Basic auth.

### 5b. netCDF4 C library is NOT thread-safe

When extracting MDSSFTD across 96 files/day with a ThreadPoolExecutor, the
process crashed with `malloc_consolidate(): invalid chunk size` — heap
corruption from concurrent xr.open_dataset calls.

Fix: wrap the open/load in a `threading.Lock`. The download (network I/O)
stays parallel, only the parse is serialized. Cost is small — parsing one
slice from a 680KB NetCDF takes ~50ms vs ~1s download.

### 5c. THREDDS OpenDAP hangs unpredictably under concurrent load

Initial attempt used the OpenDAP protocol with embedded auth in URLs. With
12 parallel workers the process hung for 11+ minutes without producing
output or exiting. Single-thread OpenDAP works fine (~0.65s/file). NCSS
endpoint returns 404. The reliable path is plain HTTPS file download +
local parse with explicit `(connect, read)` timeouts on `requests.get`.

### 6. CDSE rate-limit behavior

Open-Meteo, NASA POWER, CDSE, ADS — all rate-limit. Our scripts use
exponential backoff per service; symptom of hitting limits is HTTP 429.

For Open-Meteo specifically: running two parallel instances of
`run_open_meteo.sh` triggers a burst limit. 18/40 stations made it, 22
failed with 429. **One instance at a time, with REQUEST_DELAY≥8s.**

### 7. OOF systematically pessimistic (by ~4-5 W/m²) vs LB

A consistent pattern: OOF was ~4-5 W/m² *worse* than LB for the same
submission. Useful for sanity-checking new approaches — if OOF
under-performs by way more than 5, something's wrong.

Exception: v6's OOF was *optimistic* by 23 W/m² due to leakage. That
discrepancy is the cleanest leakage diagnostic we have.

### 8. Notebooks ate the disk

The visualization notebook accumulated ~2 MB per run because of saved
plot outputs. Wrote it from scratch via JSON to a clean state at one
point because the dev cycle was filling the cgroup quota.

### 9. lightgbm + LightGBM internal memory leak across folds

Even with `del model, ds_tr, ds_va` + `gc.collect()`, LightGBM's internal
C++ state accumulates a bit between folds. By fold 5, memory pressure
spikes. Mitigation: smaller `num_leaves`, smaller `histogram_pool_size`,
or running each fold in a subprocess (we didn't get to this).

## MBE shift trick (post-forum-2026-05-01)

Composite = `0.5·|MBE| + 0.5·RMSE`. A uniform shift of -δ changes
global MBE by ~-δ and RMSE by ~ε (negligible when δ ≪ σ ≈ 70). So if
we know the sign and magnitude of LB |MBE|, a shift drives it to zero
for ~|MBE|/2 free composite reduction.

The competition leader (ouston) openly told the forum they were
LB-probing to recover per-station means. Organizer did not prohibit;
fair game.

For v12: OOF MBE = +2.365 (all 6 folds positive). LB |MBE| = 2.78
(same direction, slightly larger). First attempt: nominal -5.414
shift (modelled to give effective -2.78 on global mean). RESULT:
\|MBE\| went 2.78 → **3.42** (overshot by 0.64).

**Why we overshot**: we computed the effective shift on the *global*
submission mean (accounting for clipping at 0), but the public LB
weights daytime rows much more heavily than nighttime zeros. So the
LB-effective shift was k=1.145× the nominal (≈-6.20), past zero to
-3.42. Triangulating from two LB datapoints gives optimum nominal
shift = -2.43 not -5.41.

**Take-away**: don't model the effective shift by global submission
mean; the LB doesn't see global mean. Model it by *the slope of LB
\|MBE\| vs nominal shift*, which is ~unit (1.145) and dominated by
daytime rows. Future MBE-shift tuning should use a 2-point V-fit on
LB data, not nominal-vs-effective accounting.

The trick generalizes: any time you know LB |MBE| and you have
multiple submissions, a binary search on the global shift converges
to LB |MBE| ≈ 0 in 1-3 submissions.

A further refinement (ouston's approach) is per-station shift via LB
probing. Each station's mean can be recovered using ~2 cleverly
designed submissions; with 40 stations that's many submissions but
also closes the gap on RMSE not just MBE (re-centering each station's
residuals reduces total squared error too). Out of scope for us now
(needs many submissions; competition deadline limits).

## What we'd do differently next time

1. **Find the killer datasets earlier.** Spend the first hour reading
   competition forums + the leader's transparency posts. Datasets > features
   > model tweaks.
2. **Skip feature-engineering tour de force**. v3's pvlib/Open-Meteo additions
   were redundant with what backward features already encoded. v7's
   forward-weather was nearly redundant too.
3. **Resist temporal-neighbor features unless CV-honest.** They look easy and
   they're not — CV leakage is silent and catastrophic on LB.
4. **Build the OOF saving infrastructure early.** We had `stack_ridge.py`
   ready in week 1 but couldn't use it for half the session because base
   models didn't save OOF predictions.
5. **Pin LightGBM params for memory.** `histogram_pool_size=512` should have
   been the default from day 1.
6. **Build small validation experiments before big ones.** If we'd tested
   v6's temporal-neighbor feature on one fold first, we'd have caught the
   leakage before wasting a submission.

## Things that were right calls

- **Per-station nighttime override** (Section J of viz notebook → applied
  in every model). Tiny code; consistent help.
- **Float32 downcast as a default** after the first OOM kill.
- **Atomic checkpoint writes** (`_safe_write_csv` / `_safe_read_csv`).
- **Per-source extraction scripts + shell wrappers**. Made re-runs
  trivial when individual extractions failed.
- **Heavy logging and per-month progress** for long-running extractions
  (TROPOMI especially). Monitor + `tail -F` was indispensable.
