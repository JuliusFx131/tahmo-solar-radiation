"""
Build a wide range of v20 ensembles + splices once the 4 Optuna-tuned standalone
models are saved. Output goes to submissions/v20_*.csv.

Combinations produced (over the 4 models {lgbm, xgb, hgb, cat} unless noted):
  • Standalone (already saved by trainer):  4 files
  • All-4 simple mean:                       1 file
  • All pairs simple mean:                   6 files
  • All triples simple mean:                 4 files
  • Per-station OOF-weighted blend:          1 file  (weight ∝ 1/per_sta_RMSE^2)
  • Rank-average of all 4:                   1 file
For each non-standalone ensemble we also save a splice variant:
  • splice TargetMBE = n150 + TargetRMSE = ensemble  ← the LB-best column trick

OOF rankings + a summary table land in submissions/v20_ensemble_summary.csv.
"""
import itertools, json, logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SUBS = ROOT / "submissions"
PROC = ROOT / "data" / "processed"
TRAIN_CSV = PROC / "Train_enhanced.csv"
SAMP = ROOT / "data" / "raw" / "SampleSubmission.csv"
ID = "ID"

MODELS = ["lgbm", "xgb", "hgb", "cat"]
N150 = SUBS / "lgbm_v12_shift_n150.csv"  # locked-in |MBE| 2.717


def per_station_metrics(pred_series, truth):
    df = truth.merge(pred_series.rename("p"), left_on=ID, right_index=True, how="inner")
    g = df.groupby("station").apply(
        lambda d: pd.Series({
            "mbe":  float(np.mean(d["p"] - d["y"])),
            "rmse": float(np.sqrt(np.mean((d["p"] - d["y"]) ** 2))),
        }), include_groups=False
    )
    return g["mbe"].abs().mean(), g["rmse"].mean(), g  # return per-station table too


def main():
    log.info("Loading truth, OOF + test predictions for v20 models ...")
    truth = pd.read_csv(TRAIN_CSV,
                       usecols=[ID, "station", "radiation (W/m2)"]
                       ).rename(columns={"radiation (W/m2)": "y"})

    missing = []
    oof = {}
    test = {}
    for m in MODELS:
        oof_path  = SUBS / f"{m}_v20_optuna_oof.csv"
        test_path = SUBS / f"{m}_v20_optuna.csv"
        if not oof_path.exists() or not test_path.exists():
            missing.append(m)
            continue
        oof[m]  = pd.read_csv(oof_path).set_index(ID)["TargetMBE"]
        test[m] = pd.read_csv(test_path).set_index(ID)["TargetMBE"]
    if missing:
        raise SystemExit(f"Missing OOF/test for: {missing}.  Run training first.")

    # Rank standalone models by per-station OOF (the actual LB metric)
    summary_rows = []
    log.info("Per-station OOF metrics per model:")
    for m in MODELS:
        mbe, rmse, _ = per_station_metrics(oof[m], truth)
        summary_rows.append({"name": m, "kind": "standalone",
                             "per_sta_abs_mbe": round(mbe, 3),
                             "per_sta_rmse":    round(rmse, 3),
                             "composite":       round(0.5*mbe + 0.5*rmse, 3)})
        log.info(f"  {m:6s}  |MBE|={mbe:.3f}  RMSE={rmse:.3f}  comp={0.5*mbe+0.5*rmse:.3f}")

    # n150 reference for splicing
    n150 = pd.read_csv(N150).set_index(ID)
    samp_ids = pd.read_csv(SAMP)[[ID]]

    def save_sub(name, test_pred_series, mbe_source=None):
        out = samp_ids.copy()
        out["TargetMBE"]  = out[ID].map(test_pred_series).clip(0, 1361)
        out["TargetRMSE"] = out[ID].map(test_pred_series).clip(0, 1361)
        path = SUBS / f"v20_{name}.csv"
        out.to_csv(path, index=False)
        # plus splice variant: n150's TargetMBE
        if mbe_source is None:
            mbe_source = n150["TargetMBE"]
        sp = samp_ids.copy()
        sp["TargetMBE"]  = sp[ID].map(mbe_source).clip(0, 1361)
        sp["TargetRMSE"] = sp[ID].map(test_pred_series).clip(0, 1361)
        splice_path = SUBS / f"v20_splice_n150mbe_{name}rmse.csv"
        sp.to_csv(splice_path, index=False)
        return path.name, splice_path.name

    # Helper to compute OOF metric for an aggregated OOF
    def record(name, kind, oof_series, member_models):
        mbe, rmse, _ = per_station_metrics(oof_series, truth)
        summary_rows.append({
            "name": name, "kind": kind,
            "members": ",".join(member_models),
            "per_sta_abs_mbe": round(mbe, 3),
            "per_sta_rmse":    round(rmse, 3),
            "composite":       round(0.5*mbe + 0.5*rmse, 3),
        })
        log.info(f"  [{kind:8s}] {name:30s}  |MBE|={mbe:.3f}  RMSE={rmse:.3f}  "
                 f"comp={0.5*mbe + 0.5*rmse:.3f}")

    # ─── All-4 simple mean ─────────────────────────────────────────
    log.info("\nAll-4 simple mean")
    oof_all = sum(oof[m] for m in MODELS) / len(MODELS)
    test_all = sum(test[m] for m in MODELS) / len(MODELS)
    record("all4_mean", "ensemble", oof_all, MODELS)
    save_sub("all4_mean", test_all)

    # ─── All pairs simple mean (6) ─────────────────────────────────
    log.info("\nAll pair-means")
    for a, b in itertools.combinations(MODELS, 2):
        name = f"pair_{a}_{b}"
        oo = (oof[a] + oof[b]) / 2
        tt = (test[a] + test[b]) / 2
        record(name, "pair", oo, [a, b])
        save_sub(name, tt)

    # ─── All triples simple mean (4) ───────────────────────────────
    log.info("\nAll triple-means")
    for combo in itertools.combinations(MODELS, 3):
        name = "triple_" + "_".join(combo)
        oo = sum(oof[m] for m in combo) / 3
        tt = sum(test[m] for m in combo) / 3
        record(name, "triple", oo, list(combo))
        save_sub(name, tt)

    # ─── OOF-weighted blend (weights ∝ 1/per_sta_rmse^2) ───────────
    rmses = []
    for m in MODELS:
        _, r, _ = per_station_metrics(oof[m], truth)
        rmses.append(r)
    inv_var = np.array([1.0 / r**2 for r in rmses])
    weights = inv_var / inv_var.sum()
    log.info(f"\nOOF-weighted blend: weights = "
             f"{dict(zip(MODELS, [round(w,3) for w in weights]))}")
    oof_w = sum(w * oof[m] for w, m in zip(weights, MODELS))
    test_w = sum(w * test[m] for w, m in zip(weights, MODELS))
    record("weighted_invrmse", "ensemble", oof_w, MODELS)
    save_sub("weighted_invrmse", test_w)

    # ─── Rank-average of the 4 ─────────────────────────────────────
    log.info("\nRank-average")
    rank_oof = sum(oof[m].rank(pct=True) for m in MODELS) / len(MODELS)
    rank_tst = sum(test[m].rank(pct=True) for m in MODELS) / len(MODELS)
    # Map percentile ranks back to a magnitude via test[lgbm] quantiles (anchored)
    anchor = test[MODELS[0]].sort_values()
    rank_tst_mag = pd.Series(
        np.interp(rank_tst.values, np.linspace(0, 1, len(anchor)), anchor.values),
        index=rank_tst.index)
    anchor_oof = oof[MODELS[0]].sort_values()
    rank_oof_mag = pd.Series(
        np.interp(rank_oof.values, np.linspace(0, 1, len(anchor_oof)), anchor_oof.values),
        index=rank_oof.index)
    record("rank_mean", "ensemble", rank_oof_mag, MODELS)
    save_sub("rank_mean", rank_tst_mag)

    # Persist the summary
    summary = pd.DataFrame(summary_rows).sort_values("composite")
    summary_path = SUBS / "v20_ensemble_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info(f"\nSaved summary: {summary_path}")
    log.info("Top 10 by composite (lower = better):")
    log.info("\n" + summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
