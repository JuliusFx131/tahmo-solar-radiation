# Models

Every training script we built, what was new in each, and the score it
achieved. All use LightGBM unless otherwise noted.

Read this top-to-bottom for the evolution of ideas, or jump to the v# you
care about.

## Common architecture

- Target: raw radiation (W/m²)
- Categorical features: `station`, `country`
- CV: 6-fold `GroupKFold` by month (except v9/v12 which become 12-month groups after pseudo-labels are added)
- 8 GB cgroup memory limit — most scripts use float32 downcast + `gc.collect()` per fold to fit
- Per-(station, hour) nighttime override applied to test predictions in every model: replace LGBM's prediction with the station's mean radiation when `ext_sol_elevation ≤ 0`

## v1 — `train_lgbm_baseline.py`

Bare-bones LGBM on the original 13-column raw data + computed solar geometry.
Time features only (hour, day-of-year, sin/cos). Per-station nighttime
override applied at inference.

**Result:** LB composite **37.72** (\|MBE\|=3.68, RMSE=71.77). Sets the floor.

## v1 companions

`train_lgbm_mae.py` (L1 objective), `train_lgbm_kt.py` (clearness-index target),
`train_lgbm_station_calib.py` (per-station linear correction on top), and
`train_lgbm_climatology.py` (per-(station, hour) climatology as a feature). Built
for ensemble diversity in the early days. KT and station_calib both **hurt LB**;
mae and clim earned a place in the v1-era 3-way ensemble (37.55).

## v2 — `train_lgbm_feat_eng_v2.py`

Added anomaly-score per station (from notebook Section K) + rolling features
of tcwv/blh + per-(station, hour) night override.

**Result:** LB 37.67. Marginal vs v1.

## v3 — `train_lgbm_feat_eng_v3.py`

First big external-data push. Added pvlib clear-sky variants
(Ineichen-Perez + Haurwitz + Linke turbidity) and Open-Meteo (GHI/DNI/DHI,
layered cloud cover, wind, dewpoint).

**Result:** LB 37.85. Marginally worse than v1 despite OOF improvement —
backward features had already captured most of the radiation signal.

## v4 — `train_lgbm_feat_eng_v4.py`

Added NASA POWER (independent radiation estimate from MERRA-2 + GEOS) and
extended CAMS aerosols to 7 species (added OC, sulphate, sea-salt + total
column water vapour). Forward-fill the 7 CAMS variables per station. Add
rolling features for NASA POWER's GHI.

**Result:** LB **37.13** (\|MBE\|=3.72, RMSE=70.53). NASA POWER's DNI ranked
above ERA5 ssrd by gain — independent radiation estimates buy diversity.

## v5 attempts — both killed

- `train_lgbm_v5_huber.py` — Huber loss with `alpha=0.9` (wrong scale for W/m²). Fold 0 RMSE=220 → killed.
- `train_lgbm_v5_log.py` — `log1p` target with bias-correction shift. Fold 0 score=47.62 → killed.

## v6 — `train_lgbm_v6_tneigh.py` — DISASTER

Added temporal-neighbour features: per (station, hour, doy) mean radiation
from training rows in surrounding ±N days.

**Result:** LB **59.93** (vs v4's 37.13). +22.8 regression.

Two flaws:
1. CV leakage — the full-train neighbor lookup let val-month rows see their
   own month's training data, inflating OOF to 37.04 (predicted LB ~33).
2. Imputation cascade for narrow-window NaN test rows created train-test
   distribution shift → systematic under-prediction (test mean 103 vs train 188).

Lesson: precompute features must respect fold boundaries, OR use as a
prediction not a feature, OR don't use them.

## v7 — `train_lgbm_v7_fwd.py`

Added forward-weather features: each row's own future temperature,
humidity, precipitation 15min/30min/1h/3h in the future. The change in
temperature over the next hour is a noisy direct measurement of current
radiation. This is an "interpolation insight" — the test set isn't the
future, it's a gap in a continuous timeline.

**Result:** LB **37.13** — tied v4. OOF improved -0.29 but backward features
had already captured most of the signal.

## v8 — `train_lgbm_v8_inversion.py` — the energy-balance push

Added two new feature families:
- **Same-day aggregates** (17 features): per (station, date) daily
  max/min/mean/std of temperature, humidity, precip; daily sum/max of
  ssrd/GHI from OM/NP/SARAH. Tells the model "today is a sunny vs cloudy day."
- **Energy-balance proxies** (8 features): dT/dt × BLH ≈ energy absorbed,
  T−dewpoint, morning warming rate, afternoon cooling rate, water vapour
  content proxy. The literal "Inversion" interpretation of the leader's team name.

**Result:** LB **36.04** (\|MBE|=3.38, RMSE=68.70). **-1.09 vs v4.** Five of top
15 features by gain came from the new families.

## v9 — `train_lgbm_v9_pseudolabel.py` — pseudo-labels v1

Take v4/v7/v8 predictions on test, select rows where std across the three
models is < 15 W/m² (high confidence), add them to training with the
consensus prediction as label (sample_weight=0.5).

Repeatedly OOM'd due to LightGBM memory accumulation across folds.
Eventually completed the original-only OOF estimate at ~38.7 across 5
folds but couldn't save the submission.

## v10 — `train_lgbm_v10_csr.py` — THE BREAKTHROUGH

Discovered (via the leader's forum thread) that
`cams-solar-radiation-timeseries` is a separate ADS dataset from the CAMS
aerosols we already had. It provides **15-min cadence pre-computed
GHI/BHI/DHI/BNI at each station's exact coordinates**, with all-sky AND
clear-sky variants + reliability flag.

Added 11 `ext_csr_*` features on top of v8.

**Result:** LB **33.03** (\|MBE\|=2.89, RMSE=63.17). **-3.01 vs v8 — the
biggest single jump of the entire session.** `ext_csr_ghi` had 10× the
gain of the next-best feature; basically a precomputed answer feature.

## v11 — `train_lgbm_v11_m2.py`

Added MERRA-2 speciated aerosols (different model than CAMS, similar info).

**Result:** OOF 37.75 (-0.16 vs v10's 37.91). Marginal — MERRA-2 mostly
redundant with the CAMS aerosols already in the model.

## v12 — `train_lgbm_v12_pseudo_v10.py` — pseudo-labels on top of v10

Same pseudo-label trick as v9 but with v10/v11/v8 as the consensus
source. v10 is so much stronger than v4 that its pseudo-labels are much
more trustworthy. Tightened std<8 W/m² + smaller LGBM (num_leaves=47,
histogram_pool_size=512) to fit memory.

125K pseudo-label rows added (18% of test).

**Result:** OOF **36.88** (-1.03 vs v10). Predicted LB ~32.0.

## v13 — `train_lgbm_v13_rfe.py` — recursive feature elimination

One-shot RFE on v10's 179 features: train once for gain ranking, drop
bottom 30% (53 features), refit 6-fold CV + full data with the lean
126-feature set. Categoricals (station, country) force-kept regardless of
rank.

Dropped families: most precip-related rolls/lags/diffs (gain near zero),
`installation_height` (constant per station, captured by station id),
raw `hour` (captured by sin/cos), `ext_csr_reliability` (low gain),
forward precip leads.

**Result:** OOF **37.83** (\|MBE\|=0.35, RMSE=75.31) vs v10's 37.91 —
essentially tied. 0.999 correlation with v10's predictions. Same LB
expectation (~33) but with meaningful memory headroom for downstream
ensembling.

## v14 — `train_lgbm_v14_pseudo_v12.py` — iterated pseudo-labels

Same recipe as v12, but the consensus source is (v12, v10, v11) instead of
(v10, v11, v8). v12 is the new strongest anchor; v8 dropped because it's
too weak to add useful disagreement signal.

The stronger consensus tightens std → **201K rows pass std<8** (29.5% of
test, vs v12's 125K = 18%). Augmented train: 843K rows.

**Result:** OOF **36.82** (\|MBE\|=2.48, RMSE=71.17) vs v12's 36.88 — flat.
Predictions 0.999-correlated with v12. The marginal pseudo rows v14 added
don't carry new information past what v12 already learned.

Lesson: pseudo-label iteration diminishes fast. The first round (v10→v12)
moved 0.41 W/m². The second (v12→v14) didn't. Stop here; the next gain
has to come from a different model class or new data, not more pseudo
rounds.

## per-station — `train_lgbm_per_station.py`

40 separate LGBM models, one per station, using v4 features.

**Result:** LB **44.08** — worse than v4 by +6.95. Per-station loses
cross-station learning that the shared model exploits.

## Stacking — `scripts/stack_ridge.py`

Ridge meta-learner on OOF predictions of multiple base models. Built but
not extensively used — needs OOF saved from each base model first.

## Ensembling

`scripts/ensemble.py` — simple weighted average of submission CSVs.

Across our entire run, **ensemble + v8/mae/clim weak models always hurt
when blended with the v10+ strong models** because the older companions
were trained on v1-era features and are now too weak to add diversity.
v10 alone beat every v8-era ensemble.
