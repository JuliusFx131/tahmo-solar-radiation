"""
v20 multi-model Optuna pipeline.

For each of {lgbm, xgb, hgb, cat}:
  STEP 1.  Optuna hyperparam search on a 2-fold (month-grouped) holdout — clean
           train rows only, no pseudo-labels. RMSE objective. N trials configurable
           per model.
  STEP 2.  Generate consensus pseudo-labels for high-confidence test rows
           (mean of n150 + v16 + v18; std < PSEUDO_STD_THRESH; daytime only).
  STEP 3.  Refit best params on (train + pseudo) with 6-fold GroupKFold-by-month CV.
           Score on original-train rows only for honest OOF. Save OOF csv.
  STEP 4.  Refit on the full augmented set for mean(best_iter) rounds.
           Predict test. Apply per-(station,hour) night override. Save submission.

Run:    python3.10 scripts/train_v20_optuna.py [lgbm|xgb|hgb|cat|all]
Outputs (per model):
  submissions/{model}_v20_optuna.csv
  submissions/{model}_v20_optuna_oof.csv
  submissions/{model}_v20_optuna_log.json
"""
import gc, json, logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import GroupKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.INFO)  # show trial results in log

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
SUBS = ROOT / "submissions"
SUBS.mkdir(parents=True, exist_ok=True)
SAMP = ROOT / "data" / "raw" / "SampleSubmission.csv"
TRAIN_PARQ = PROC / "v20_train.parquet"
TEST_PARQ  = PROC / "v20_test.parquet"
FEATS_JSON = PROC / "v20_feature_list.json"
TRAIN_CSV  = PROC / "Train_enhanced.csv"   # for per-(station,hour) night override

TARGET = "radiation (W/m2)"
ID     = "ID"

# Pseudo-label config
PSEUDO_SOURCES = [
    SUBS / "lgbm_v12_shift_n150.csv",
    SUBS / "lgbm_v16_mdssftd.csv",
    SUBS / "lgbm_v18_mdss_pseudo.csv",
]
PSEUDO_STD_THRESH = 8.0
PSEUDO_WEIGHT     = 0.5
DAYTIME_ELEV_THR  = 5.0

# Optuna trial counts per model (set conservatively for an overnight run)
TRIALS = {"lgbm": 20, "xgb": 20, "hgb": 20, "cat": 12}
SEED = 42


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers

def load_data():
    train = pd.read_parquet(TRAIN_PARQ)
    test  = pd.read_parquet(TEST_PARQ)
    meta  = json.loads(FEATS_JSON.read_text())
    features = meta["features"]
    categorical = meta["categorical"]
    # Ensure categorical dtype survived the parquet round-trip
    for col in categorical:
        if not isinstance(train[col].dtype, pd.CategoricalDtype):
            cats = sorted(set(train[col].astype(str)) | set(test[col].astype(str)))
            train[col] = pd.Categorical(train[col].astype(str), categories=cats)
            test[col]  = pd.Categorical(test[col].astype(str),  categories=cats)
    log.info(f"  features={len(features)}  train={len(train):,}  test={len(test):,}")
    return train, test, features, categorical


def build_pseudo(test):
    log.info("Building pseudo-label set from existing submissions ...")
    preds = []
    for p in PSEUDO_SOURCES:
        if not p.exists():
            log.warning(f"  {p.name} missing — pseudo confidence proxy will use fewer sources")
            continue
        preds.append(pd.read_csv(p).set_index(ID)["TargetMBE"].rename(p.stem))
    if len(preds) < 2:
        raise RuntimeError("Need >=2 source subs for pseudo-labels")
    pred_df = pd.concat(preds, axis=1)
    pred_df["pred_mean"] = pred_df.mean(axis=1)
    pred_df["pred_std"]  = pred_df.std(axis=1)
    pred_df = pred_df.reset_index()
    merged = test[[ID, "ext_sol_elevation"]].merge(pred_df, on=ID, how="left")
    confident = (
        (merged["pred_std"] < PSEUDO_STD_THRESH)
        & (merged["ext_sol_elevation"] > DAYTIME_ELEV_THR)
        & merged["pred_mean"].notna()
    )
    log.info(f"  pseudo-label candidates (daytime, std<{PSEUDO_STD_THRESH}): "
             f"{confident.sum():,} / {len(merged):,} "
             f"({confident.mean()*100:.1f}%)")
    pseudo = merged.loc[confident, [ID, "pred_mean"]].rename(columns={"pred_mean": TARGET})
    return pseudo


def score_components(y_true, y_pred):
    mbe = float(np.mean(y_pred - y_true))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return abs(mbe), rmse, 0.5*abs(mbe) + 0.5*rmse


def build_night_override():
    src = pd.read_csv(TRAIN_CSV,
                      usecols=["station","timestamp","ext_sol_elevation",TARGET],
                      parse_dates=["timestamp"])
    src = src[src["ext_sol_elevation"] <= 0].copy()
    src["hour"] = src["timestamp"].dt.hour
    per_hour = src.groupby(["station","hour"])[TARGET].mean().to_dict()
    per_sta  = src.groupby("station")[TARGET].mean().to_dict()
    fallback = float(np.mean(list(per_sta.values())) if per_sta else 0.0)
    return per_hour, per_sta, fallback


def apply_night_override(test_meta, test_pred, per_hour, per_sta, fallback):
    is_night = test_meta["ext_sol_elevation"].values <= 0
    hours = test_meta["timestamp"].dt.hour.values
    stations = test_meta["station"].astype(str).values
    night_vals = np.array([per_hour.get((s,h), per_sta.get(s, fallback))
                           for s,h in zip(stations, hours)])
    out = np.where(is_night, night_vals, test_pred)
    return np.clip(out, 0, 1361)


def build_aug_frame_inplace(train, test, pseudo, features, cat_features):
    """
    Build the augmented DataFrame (train + confident-test pseudo rows) by
    pre-allocating one numpy array per column and filling each [train|pseudo]
    slice once.  Avoids the pd.concat OOM peak where train + test_sub + concat
    result + buffer copies all live simultaneously.

    Peak memory ≈ train + test_sub + aug + one column-buffer at a time.
    """
    pseudo_target = pseudo.set_index(ID)[TARGET].astype(np.float32)
    keep_mask = test[ID].isin(pseudo_target.index).values
    n_train, n_pseudo = len(train), int(keep_mask.sum())
    n_total = n_train + n_pseudo
    test_sub = test.loc[keep_mask]
    log.info(f"  building aug frame: n_train={n_train:,} + n_pseudo={n_pseudo:,} "
             f"= {n_total:,}")

    out = {}
    for col in features + [TARGET, ID, "timestamp"]:
        if col == TARGET:
            arr = np.empty(n_total, dtype=np.float32)
            arr[:n_train] = train[TARGET].values.astype(np.float32, copy=False)
            arr[n_train:] = test_sub[ID].map(pseudo_target).astype(np.float32).values
            out[col] = arr
        elif col == ID:
            arr = np.empty(n_total, dtype=object)
            arr[:n_train] = train[ID].values
            arr[n_train:] = test_sub[ID].values
            out[col] = arr
        elif col == "timestamp":
            arr = np.empty(n_total, dtype="datetime64[ns]")
            arr[:n_train] = train["timestamp"].values
            arr[n_train:] = test_sub["timestamp"].values
            out[col] = arr
        elif col in cat_features:
            cats = train[col].cat.categories
            arr = np.empty(n_total, dtype=object)
            arr[:n_train] = train[col].astype(str).values
            arr[n_train:] = test_sub[col].astype(str).values
            out[col] = pd.Categorical(arr, categories=cats)
            del arr
        else:
            arr = np.empty(n_total, dtype=np.float32)
            arr[:n_train] = train[col].values.astype(np.float32, copy=False)
            arr[n_train:] = test_sub[col].values.astype(np.float32, copy=False)
            out[col] = arr

    origin = np.empty(n_total, dtype=object)
    origin[:n_train] = "train"
    origin[n_train:] = "pseudo"
    out["_origin"] = origin

    aug = pd.DataFrame(out)
    del out, test_sub, pseudo_target; gc.collect()
    aug = aug.sort_values(["station", "timestamp"]).reset_index(drop=True) \
        if "station" in aug.columns else aug.sort_values(["timestamp"]).reset_index(drop=True)
    return aug


def to_float32_numpy(df, features, cat_features):
    """
    Pack DataFrame columns into a single contiguous float32 2D numpy array.
    Categorical columns are stored as their integer codes cast to float32
    (HGB and LGBM accept this; XGB treats them as ordinal which is fine for
    low-cardinality station/country).  Avoids the pandas .values upcast to
    float64 when dtypes are heterogeneous.
    """
    X = np.empty((len(df), len(features)), dtype=np.float32)
    for j, col in enumerate(features):
        if col in cat_features:
            X[:, j] = df[col].cat.codes.values.astype(np.float32)
        else:
            X[:, j] = df[col].values.astype(np.float32, copy=False)
    return X


def save_outputs(model_name, oof_pred, train_ids, test_pred, test_ids, best_params,
                 cv_folds, oof_score, oof_mbe, oof_rmse, wall_s, n_pseudo):
    tag = f"{model_name}_v20_optuna"
    sub_path = SUBS / f"{tag}.csv"
    oof_path = SUBS / f"{tag}_oof.csv"
    log_path = SUBS / f"{tag}_log.json"

    # OOF csv
    pd.DataFrame({
        ID: train_ids,
        "TargetMBE":  np.clip(oof_pred, 0, 1361),
        "TargetRMSE": np.clip(oof_pred, 0, 1361),
    }).to_csv(oof_path, index=False)

    # Submission (re-aligned to sample)
    sub = pd.DataFrame({ID: test_ids, "TargetMBE": test_pred, "TargetRMSE": test_pred})
    sample = pd.read_csv(SAMP)
    sub = sample[[ID]].merge(sub, on=ID, how="left")
    sub.to_csv(sub_path, index=False)

    log_path.write_text(json.dumps({
        "model":       model_name,
        "tag":         tag,
        "best_params": best_params,
        "cv_folds":    cv_folds,
        "oof_score":   round(oof_score, 3),
        "oof_mbe":     round(oof_mbe, 3),
        "oof_rmse":    round(oof_rmse, 3),
        "n_pseudo":    int(n_pseudo),
        "wall_time_s": round(wall_s, 1),
    }, indent=2, default=str))
    log.info(f"  saved {sub_path.name}, {oof_path.name}, {log_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Per-model implementations: optuna_search(...) + refit_and_predict(...)

def run_lgbm(features, cat_features):
    import lightgbm as lgb
    t0 = time.time()
    log.info("=" * 60)
    log.info("[LGBM] starting Optuna + pseudo-label refit")

    train, test, _, _ = load_data()
    y_orig = train[TARGET].values.astype(np.float32)
    groups = train["timestamp"].dt.month.values

    # 2-fold Optuna holdout: months 2 (Feb) + 11 (Nov) — historically hardest
    val_months = [2, 11]
    tr_mask = ~pd.Series(groups).isin(val_months).values
    va_mask = ~tr_mask
    log.info(f"  optuna val rows: {va_mask.sum():,}  train rows: {tr_mask.sum():,}")

    # Pre-build the two Datasets once and reuse across trials.  Force eager
    # construct() so LGBM copies into its binary format, then drop the source
    # DataFrame for the Optuna phase — keeps memory <4 GB during search.
    ds_tr_search = lgb.Dataset(train[features].iloc[tr_mask.nonzero()[0]],
                               label=y_orig[tr_mask],
                               categorical_feature=cat_features)
    ds_va_search = lgb.Dataset(train[features].iloc[va_mask.nonzero()[0]],
                               label=y_orig[va_mask],
                               categorical_feature=cat_features,
                               reference=ds_tr_search)
    ds_tr_search.construct(); ds_va_search.construct()
    del train, test, y_orig, groups, tr_mask, va_mask; gc.collect()

    def objective(trial):
        params = dict(
            objective="regression", metric="rmse",
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 127),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 50, 300),
            feature_fraction=trial.suggest_float("feature_fraction", 0.6, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0),
            bagging_freq=5,
            lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            verbose=-1, n_jobs=-1, seed=SEED,
        )
        m = lgb.train(params, ds_tr_search, num_boost_round=1500, valid_sets=[ds_va_search],
                      callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
        rmse = m.best_score["valid_0"]["rmse"]
        del m; gc.collect()
        return rmse

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=TRIALS["lgbm"], show_progress_bar=False)
    best = dict(study.best_params)
    best.update(objective="regression", metric="rmse", bagging_freq=5,
                verbose=-1, n_jobs=-1, seed=SEED)
    log.info(f"[LGBM] best RMSE on holdout: {study.best_value:.3f}")
    log.info(f"[LGBM] best params: {study.best_params}")
    del ds_tr_search, ds_va_search, study; gc.collect()

    # --- Step 2: pseudo-labels (memory-safe in-place build) ---
    train, test, _, _ = load_data()
    pseudo = build_pseudo(test)
    aug = build_aug_frame_inplace(train, test, pseudo, features, cat_features)
    del train, pseudo; gc.collect()
    y_aug = aug[TARGET].values
    w_aug = np.where(aug["_origin"].values == "pseudo",
                     np.float32(PSEUDO_WEIGHT), np.float32(1.0))
    origin_arr = aug["_origin"].values.copy()
    groups_aug = aug["timestamp"].dt.month.values
    orig_train_ids = aug.loc[aug["_origin"] == "train", ID].values.copy()
    n_pseudo = int((origin_arr == "pseudo").sum())
    X_aug = aug[features]   # no .copy(); fold slices materialise per Dataset

    # --- Step 3: refit best params, 6-fold CV ---
    log.info("[LGBM] refit with best params, 6-fold CV ...")
    gkf = GroupKFold(n_splits=6)
    oof_aug = np.zeros(len(y_aug), dtype=np.float32)
    fold_meta = []
    best_iters = []
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_aug, y_aug, groups_aug)):
        ds_tr = lgb.Dataset(X_aug.iloc[tr_idx], label=y_aug[tr_idx],
                            weight=w_aug[tr_idx], categorical_feature=cat_features)
        ds_va = lgb.Dataset(X_aug.iloc[va_idx], label=y_aug[va_idx],
                            weight=w_aug[va_idx], categorical_feature=cat_features)
        m = lgb.train(best, ds_tr, num_boost_round=2500, valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
        oof_aug[va_idx] = m.predict(X_aug.iloc[va_idx]).astype(np.float32)
        orig_mask = origin_arr[va_idx] == "train"
        if orig_mask.sum() > 0:
            mbe, rmse, sc = score_components(y_aug[va_idx][orig_mask],
                                              oof_aug[va_idx][orig_mask])
            log.info(f"  fold {fold_idx}: |MBE|={mbe:.2f} RMSE={rmse:.2f} "
                     f"score={sc:.2f} best={m.best_iteration}")
            fold_meta.append({"fold":fold_idx, "mbe":mbe, "rmse":rmse, "score":sc,
                              "best_iter":m.best_iteration})
            best_iters.append(m.best_iteration)
        del ds_tr, ds_va, m; gc.collect()

    orig_mask_all = (origin_arr == "train")
    oof_orig = oof_aug[orig_mask_all]
    mbe_a, rmse_a, sc_a = score_components(y_aug[orig_mask_all], oof_orig)
    log.info(f"[LGBM] OOF orig: |MBE|={mbe_a:.2f} RMSE={rmse_a:.2f} score={sc_a:.2f}")

    # --- Step 4: refit on full aug, predict test ---
    best_iter = int(np.mean(best_iters))
    log.info(f"[LGBM] final refit on full aug for {best_iter} rounds ...")
    full_ds = lgb.Dataset(X_aug, label=y_aug, weight=w_aug, categorical_feature=cat_features)
    final_model = lgb.train(best, full_ds, num_boost_round=best_iter)
    del full_ds, X_aug, y_aug, w_aug, groups_aug; gc.collect()

    test_pred = final_model.predict(test[features]).astype(np.float32)
    del final_model; gc.collect()
    per_hour, per_sta, fb = build_night_override()
    test_pred = apply_night_override(test, test_pred, per_hour, per_sta, fb)

    save_outputs("lgbm", oof_orig, orig_train_ids, test_pred, test[ID].values,
                 best, fold_meta, sc_a, mbe_a, rmse_a, time.time()-t0, n_pseudo)
    del test; gc.collect()


def run_xgb(features, cat_features):
    import xgboost as xgb
    t0 = time.time()
    log.info("=" * 60)
    log.info("[XGB] starting Optuna + pseudo-label refit")
    train, test, _, _ = load_data()
    y = train[TARGET].values.astype(np.float32)
    groups = train["timestamp"].dt.month.values
    val_months = [2, 11]
    tr_mask = ~pd.Series(groups).isin(val_months).values
    va_mask = ~tr_mask

    dtr = xgb.DMatrix(train[features].iloc[tr_mask.nonzero()[0]],
                      label=y[tr_mask], enable_categorical=True)
    dva = xgb.DMatrix(train[features].iloc[va_mask.nonzero()[0]],
                      label=y[va_mask], enable_categorical=True)
    # DMatrix has internalised the data; free the DataFrames during the
    # Optuna phase to keep RSS under control.  Reloaded for pseudo-aug below.
    del train, test, y, groups, tr_mask, va_mask; gc.collect()

    def objective(trial):
        params = dict(
            objective="reg:squarederror", eval_metric="rmse",
            tree_method="hist",
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            max_depth=trial.suggest_int("max_depth", 5, 10),
            min_child_weight=trial.suggest_int("min_child_weight", 10, 200),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            nthread=-1, seed=SEED,
        )
        m = xgb.train(params, dtr, num_boost_round=1500,
                      evals=[(dva, "val")], early_stopping_rounds=40, verbose_eval=False)
        score = float(m.best_score)
        del m; gc.collect()
        return score

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=TRIALS["xgb"], show_progress_bar=False)
    best = dict(study.best_params)
    best.update(objective="reg:squarederror", eval_metric="rmse",
                tree_method="hist", nthread=-1, seed=SEED)
    log.info(f"[XGB] best RMSE on holdout: {study.best_value:.3f}")
    log.info(f"[XGB] best params: {study.best_params}")
    del dtr, dva, study; gc.collect()

    train, test, _, _ = load_data()
    pseudo = build_pseudo(test)
    aug = build_aug_frame_inplace(train, test, pseudo, features, cat_features)
    del train, pseudo; gc.collect()
    y_aug = aug[TARGET].values
    w_aug = np.where(aug["_origin"].values == "pseudo",
                     np.float32(PSEUDO_WEIGHT), np.float32(1.0))
    origin_arr = aug["_origin"].values.copy()
    groups_aug = aug["timestamp"].dt.month.values
    orig_train_ids = aug.loc[aug["_origin"] == "train", ID].values.copy()
    n_pseudo = int((origin_arr == "pseudo").sum())
    X_aug = aug[features]

    log.info("[XGB] refit best params, 6-fold CV ...")
    gkf = GroupKFold(n_splits=6)
    oof_aug = np.zeros(len(y_aug), dtype=np.float32)
    fold_meta = []; best_iters = []
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_aug, y_aug, groups_aug)):
        dtr = xgb.DMatrix(X_aug.iloc[tr_idx], label=y_aug[tr_idx],
                          weight=w_aug[tr_idx], enable_categorical=True)
        dva = xgb.DMatrix(X_aug.iloc[va_idx], label=y_aug[va_idx],
                          weight=w_aug[va_idx], enable_categorical=True)
        m = xgb.train(best, dtr, num_boost_round=2500,
                      evals=[(dva,"val")], early_stopping_rounds=40, verbose_eval=False)
        oof_aug[va_idx] = m.predict(dva).astype(np.float32)
        orig_mask = origin_arr[va_idx] == "train"
        if orig_mask.sum() > 0:
            mbe, rmse, sc = score_components(y_aug[va_idx][orig_mask],
                                              oof_aug[va_idx][orig_mask])
            log.info(f"  fold {fold_idx}: |MBE|={mbe:.2f} RMSE={rmse:.2f} "
                     f"score={sc:.2f} best={m.best_iteration}")
            fold_meta.append({"fold":fold_idx,"mbe":mbe,"rmse":rmse,"score":sc,
                              "best_iter":m.best_iteration})
            best_iters.append(m.best_iteration)
        del dtr, dva, m; gc.collect()

    orig_mask_all = origin_arr == "train"
    oof_orig = oof_aug[orig_mask_all]
    mbe_a, rmse_a, sc_a = score_components(y_aug[orig_mask_all], oof_orig)
    log.info(f"[XGB] OOF orig: |MBE|={mbe_a:.2f} RMSE={rmse_a:.2f} score={sc_a:.2f}")

    best_iter = int(np.mean(best_iters))
    log.info(f"[XGB] final refit on full aug for {best_iter} rounds ...")
    dfull = xgb.DMatrix(X_aug, label=y_aug, weight=w_aug, enable_categorical=True)
    final = xgb.train(best, dfull, num_boost_round=best_iter)
    dtest = xgb.DMatrix(test[features], enable_categorical=True)
    test_pred = final.predict(dtest).astype(np.float32)
    per_hour, per_sta, fb = build_night_override()
    test_pred = apply_night_override(test, test_pred, per_hour, per_sta, fb)

    save_outputs("xgb", oof_orig, orig_train_ids, test_pred, test[ID].values,
                 best, fold_meta, sc_a, mbe_a, rmse_a, time.time()-t0, n_pseudo)
    del test, dtest, final; gc.collect()


def run_hgb(features, cat_features):
    from sklearn.ensemble import HistGradientBoostingRegressor
    t0 = time.time()
    log.info("=" * 60)
    log.info("[HGB] starting Optuna + pseudo-label refit")
    train, test, _, _ = load_data()
    cat_idx = [features.index(c) for c in cat_features]
    X_full = to_float32_numpy(train, features, cat_features)
    y = train[TARGET].values.astype(np.float32)
    groups = train["timestamp"].dt.month.values
    val_months = [2, 11]
    tr_mask = ~pd.Series(groups).isin(val_months).values
    va_mask = ~tr_mask
    Xtr = X_full[tr_mask.nonzero()[0]]
    Xva = X_full[va_mask.nonzero()[0]]
    ytr = y[tr_mask]; yva = y[va_mask]
    del train, test, X_full, y, groups, tr_mask, va_mask; gc.collect()

    def objective(trial):
        params = dict(
            loss="squared_error",
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 31, 127),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 50, 300),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            max_iter=1500, early_stopping=False,
            categorical_features=cat_idx, random_state=SEED,
        )
        m = HistGradientBoostingRegressor(**params)
        m.fit(Xtr, ytr)
        best_rmse = float("inf")
        for pred in m.staged_predict(Xva):
            rmse = float(np.sqrt(np.mean((pred - yva) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
        del m; gc.collect()
        return best_rmse

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=TRIALS["hgb"], show_progress_bar=False)
    best = dict(study.best_params)
    log.info(f"[HGB] best RMSE on holdout: {study.best_value:.3f}")
    log.info(f"[HGB] best params: {best}")
    del Xtr, Xva, ytr, yva, study; gc.collect()

    train, test, _, _ = load_data()

    pseudo = build_pseudo(test)
    aug = build_aug_frame_inplace(train, test, pseudo, features, cat_features)
    del train, pseudo; gc.collect()
    y_aug = aug[TARGET].values
    w_aug = np.where(aug["_origin"].values == "pseudo",
                     np.float32(PSEUDO_WEIGHT), np.float32(1.0))
    origin_arr = aug["_origin"].values.copy()
    groups_aug = aug["timestamp"].dt.month.values
    orig_train_ids = aug.loc[aug["_origin"] == "train", ID].values.copy()
    n_pseudo = int((origin_arr == "pseudo").sum())
    X_aug_np = to_float32_numpy(aug, features, cat_features)
    del aug; gc.collect()

    log.info("[HGB] refit best params, 6-fold CV ...")
    gkf = GroupKFold(n_splits=6)
    oof_aug = np.zeros(len(y_aug), dtype=np.float32)
    fold_meta = []; best_iters = []
    refit_params = dict(loss="squared_error",
                        learning_rate=best["learning_rate"],
                        max_leaf_nodes=best["max_leaf_nodes"],
                        min_samples_leaf=best["min_samples_leaf"],
                        l2_regularization=best["l2_regularization"],
                        max_iter=2500, early_stopping=False,
                        categorical_features=cat_idx, random_state=SEED)
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_aug_np, y_aug, groups_aug)):
        m = HistGradientBoostingRegressor(**refit_params)
        m.fit(X_aug_np[tr_idx], y_aug[tr_idx], sample_weight=w_aug[tr_idx])
        # find best iter via staged_predict
        best_rmse, best_iter, best_pred = float("inf"), 0, None
        for it, pred in enumerate(m.staged_predict(X_aug_np[va_idx])):
            rmse = float(np.sqrt(np.mean((pred - y_aug[va_idx])**2)))
            if rmse < best_rmse:
                best_rmse, best_iter, best_pred = rmse, it+1, pred
        oof_aug[va_idx] = best_pred.astype(np.float32)
        orig_mask = origin_arr[va_idx] == "train"
        if orig_mask.sum() > 0:
            mbe, rmse, sc = score_components(y_aug[va_idx][orig_mask],
                                              oof_aug[va_idx][orig_mask])
            log.info(f"  fold {fold_idx}: |MBE|={mbe:.2f} RMSE={rmse:.2f} "
                     f"score={sc:.2f} best={best_iter}")
            fold_meta.append({"fold":fold_idx,"mbe":mbe,"rmse":rmse,"score":sc,
                              "best_iter":best_iter})
            best_iters.append(best_iter)
        del m; gc.collect()

    orig_mask_all = origin_arr == "train"
    oof_orig = oof_aug[orig_mask_all]
    mbe_a, rmse_a, sc_a = score_components(y_aug[orig_mask_all], oof_orig)
    log.info(f"[HGB] OOF orig: |MBE|={mbe_a:.2f} RMSE={rmse_a:.2f} score={sc_a:.2f}")

    best_iter = int(np.mean(best_iters))
    log.info(f"[HGB] final refit on full aug for {best_iter} iters ...")
    refit_params["max_iter"] = best_iter
    final = HistGradientBoostingRegressor(**refit_params)
    final.fit(X_aug_np, y_aug, sample_weight=w_aug)
    del X_aug_np, y_aug, w_aug; gc.collect()
    Xtest_np = to_float32_numpy(test, features, cat_features)
    test_pred = final.predict(Xtest_np).astype(np.float32)
    del Xtest_np; gc.collect()
    per_hour, per_sta, fb = build_night_override()
    test_pred = apply_night_override(test, test_pred, per_hour, per_sta, fb)

    save_outputs("hgb", oof_orig, orig_train_ids, test_pred, test[ID].values,
                 best, fold_meta, sc_a, mbe_a, rmse_a, time.time()-t0, n_pseudo)
    del test, final; gc.collect()


def run_cat(features, cat_features):
    from catboost import CatBoostRegressor, Pool
    t0 = time.time()
    log.info("=" * 60)
    log.info("[CAT] starting Optuna + pseudo-label refit")
    train, test, _, _ = load_data()
    y = train[TARGET].values.astype(np.float32)
    groups = train["timestamp"].dt.month.values
    val_months = [2, 11]
    tr_mask = ~pd.Series(groups).isin(val_months).values
    va_mask = ~tr_mask
    cat_idx = [features.index(c) for c in cat_features]
    X = train[features].copy()
    for c in cat_features:
        X[c] = X[c].astype(str)
    tr_pool = Pool(X.iloc[tr_mask.nonzero()[0]], label=y[tr_mask], cat_features=cat_idx)
    va_pool = Pool(X.iloc[va_mask.nonzero()[0]], label=y[va_mask], cat_features=cat_idx)
    # Pool has internalised the data — free the DataFrames during search.
    del X, train, test, y, groups, tr_mask, va_mask; gc.collect()

    def objective(trial):
        params = dict(
            loss_function="RMSE",
            iterations=1500,
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            depth=trial.suggest_int("depth", 5, 9),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            random_strength=trial.suggest_float("random_strength", 0.5, 5.0),
            border_count=trial.suggest_int("border_count", 64, 254),
            random_seed=SEED, thread_count=-1, verbose=False,
            early_stopping_rounds=40,
        )
        m = CatBoostRegressor(**params)
        m.fit(tr_pool, eval_set=va_pool)
        score = float(m.best_score_["validation"]["RMSE"])
        del m; gc.collect()
        return score

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=TRIALS["cat"], show_progress_bar=False)
    best = dict(study.best_params)
    log.info(f"[CAT] best RMSE on holdout: {study.best_value:.3f}")
    log.info(f"[CAT] best params: {best}")
    del tr_pool, va_pool, study; gc.collect()

    train, test, _, _ = load_data()
    pseudo = build_pseudo(test)
    aug = build_aug_frame_inplace(train, test, pseudo, features, cat_features)
    del train, pseudo; gc.collect()
    y_aug = aug[TARGET].values
    w_aug = np.where(aug["_origin"].values == "pseudo",
                     np.float32(PSEUDO_WEIGHT), np.float32(1.0))
    origin_arr = aug["_origin"].values.copy()
    groups_aug = aug["timestamp"].dt.month.values
    orig_train_ids = aug.loc[aug["_origin"] == "train", ID].values.copy()
    n_pseudo = int((origin_arr == "pseudo").sum())
    X_aug = aug[features].copy()
    for c in cat_features:
        X_aug[c] = X_aug[c].astype(str)
    del aug; gc.collect()

    log.info("[CAT] refit best params, 6-fold CV ...")
    gkf = GroupKFold(n_splits=6)
    oof_aug = np.zeros(len(y_aug), dtype=np.float32)
    fold_meta = []; best_iters = []
    refit_params = dict(loss_function="RMSE", iterations=2500,
                        learning_rate=best["learning_rate"],
                        depth=best["depth"],
                        l2_leaf_reg=best["l2_leaf_reg"],
                        random_strength=best["random_strength"],
                        border_count=best["border_count"],
                        random_seed=SEED, thread_count=-1, verbose=False,
                        early_stopping_rounds=40)
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_aug, y_aug, groups_aug)):
        tr_pool = Pool(X_aug.iloc[tr_idx], label=y_aug[tr_idx],
                       weight=w_aug[tr_idx], cat_features=cat_idx)
        va_pool = Pool(X_aug.iloc[va_idx], label=y_aug[va_idx],
                       weight=w_aug[va_idx], cat_features=cat_idx)
        m = CatBoostRegressor(**refit_params)
        m.fit(tr_pool, eval_set=va_pool)
        oof_aug[va_idx] = m.predict(va_pool).astype(np.float32)
        orig_mask = origin_arr[va_idx] == "train"
        if orig_mask.sum() > 0:
            mbe, rmse, sc = score_components(y_aug[va_idx][orig_mask],
                                              oof_aug[va_idx][orig_mask])
            log.info(f"  fold {fold_idx}: |MBE|={mbe:.2f} RMSE={rmse:.2f} "
                     f"score={sc:.2f} best={m.best_iteration_}")
            fold_meta.append({"fold":fold_idx,"mbe":mbe,"rmse":rmse,"score":sc,
                              "best_iter":m.best_iteration_})
            best_iters.append(m.best_iteration_)
        del tr_pool, va_pool, m; gc.collect()

    orig_mask_all = origin_arr == "train"
    oof_orig = oof_aug[orig_mask_all]
    mbe_a, rmse_a, sc_a = score_components(y_aug[orig_mask_all], oof_orig)
    log.info(f"[CAT] OOF orig: |MBE|={mbe_a:.2f} RMSE={rmse_a:.2f} score={sc_a:.2f}")

    best_iter = int(np.mean(best_iters))
    log.info(f"[CAT] final refit on full aug for {best_iter} iters ...")
    refit_params["iterations"] = best_iter
    refit_params.pop("early_stopping_rounds", None)
    full_pool = Pool(X_aug, label=y_aug, weight=w_aug, cat_features=cat_idx)
    final = CatBoostRegressor(**refit_params)
    final.fit(full_pool)
    Xtest = test[features].copy()
    for c in cat_features:
        Xtest[c] = Xtest[c].astype(str)
    test_pool = Pool(Xtest, cat_features=cat_idx)
    test_pred = final.predict(test_pool).astype(np.float32)
    per_hour, per_sta, fb = build_night_override()
    test_pred = apply_night_override(test, test_pred, per_hour, per_sta, fb)

    save_outputs("cat", oof_orig, orig_train_ids, test_pred, test[ID].values,
                 best, fold_meta, sc_a, mbe_a, rmse_a, time.time()-t0, n_pseudo)


# ──────────────────────────────────────────────────────────────────────────────

RUNNERS = {"lgbm": run_lgbm, "xgb": run_xgb, "hgb": run_hgb, "cat": run_cat}


def main():
    which = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    log.info(f"Run target: {which}")
    # Each runner reloads parquets internally to guarantee fresh memory state
    # — avoids OOM when running all 4 in sequence under the 8 GB cgroup.
    train_peek, _, features, cat_features = load_data()
    del train_peek; gc.collect()

    targets = ["lgbm","xgb","hgb","cat"] if which == "all" else [which]
    for t in targets:
        if t not in RUNNERS:
            log.error(f"unknown model: {t}");  continue
        # Skip if this model's outputs already exist (allows resume after crash)
        sub_path = SUBS / f"{t}_v20_optuna.csv"
        if sub_path.exists():
            log.info(f"[{t.upper()}] outputs already exist at {sub_path}, skipping")
            continue
        RUNNERS[t](features, cat_features)
        gc.collect()

    log.info("DONE.")


if __name__ == "__main__":
    main()
