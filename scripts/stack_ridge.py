"""
Ridge-regression stacking meta-learner.

Takes the OOF and test predictions from several base models, fits a Ridge
regression on the OOF stack to learn the optimal blend (vs. our previous
equal/manual weighting), and applies the same coefficients to the test
predictions.

The OOF predictions come from the run-log JSON of each base model — we
expect them at fixed paths and read them.

Inputs (per base model):
  /workspace/submissions/<tag>.csv           — test predictions (one column)
  /workspace/submissions/<tag>_oof.csv       — OOF predictions (one column)

Currently base models we know how to stack:
  feat_eng_v4   (RMSE-optimised LGBM, current single-model LB winner)
  v6_tneigh     (v4 + temporal neighbours, expected stronger)
  mae_v1        (L1 objective)
  climatology_v1

Output:
  /workspace/submissions/stack_ridge_v1.csv
  /workspace/submissions/stack_ridge_v1_log.txt
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SUBS = ROOT / "submissions"
PROC = ROOT / "data" / "processed"

TRAIN_CSV = ROOT / "data" / "raw" / "Train.csv"
SAMP      = ROOT / "data" / "raw" / "SampleSubmission.csv"


def score_components(y_true, y_pred):
    mbe  = float(np.mean(y_pred - y_true))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return abs(mbe), rmse, 0.5 * abs(mbe) + 0.5 * rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", required=True,
                    help="Base tags. Each needs <tag>.csv and <tag>_oof.csv in /submissions/")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Ridge regularisation strength")
    ap.add_argument("--out", default="stack_ridge_v1",
                    help="Output tag (no extension)")
    args = ap.parse_args()

    # Load OOF radiation per base model + true train radiation
    log.info("Loading true train radiation ...")
    train = pd.read_csv(TRAIN_CSV, usecols=["ID", "radiation (W/m2)"])
    y_true = train.set_index("ID")["radiation (W/m2)"].values

    # OOF stack
    oof_cols, test_cols = {}, {}
    for tag in args.bases:
        oof_path = SUBS / f"{tag}_oof.csv"
        test_path = SUBS / f"{tag}.csv"
        if not oof_path.exists():
            log.error(f"  {oof_path} missing — re-run that script with OOF saving on")
            return
        if not test_path.exists():
            log.error(f"  {test_path} missing")
            return
        oof = pd.read_csv(oof_path).set_index("ID")
        sub = pd.read_csv(test_path).set_index("ID")
        # Both files have TargetMBE column (== TargetRMSE)
        oof_cols[tag]  = oof.reindex(train["ID"])["TargetMBE"].values
        test_cols[tag] = sub["TargetMBE"]
        oof_score = score_components(y_true, oof_cols[tag])
        log.info(f"  {tag}: OOF |MBE|={oof_score[0]:.2f}  RMSE={oof_score[1]:.2f}  "
                 f"score={oof_score[2]:.2f}")

    X_oof = np.column_stack([oof_cols[t] for t in args.bases])

    log.info(f"Fitting Ridge (alpha={args.alpha}) on OOF stack ...")
    ridge = Ridge(alpha=args.alpha, fit_intercept=True)
    ridge.fit(X_oof, y_true)
    log.info(f"  intercept: {ridge.intercept_:+.3f}")
    for t, c in zip(args.bases, ridge.coef_):
        log.info(f"  weight {t}: {c:+.4f}")

    stack_oof_pred = ridge.predict(X_oof)
    stack_oof_pred = np.clip(stack_oof_pred, 0, 1361)
    stack_oof_score = score_components(y_true, stack_oof_pred)
    log.info(f"Stacked OOF:  |MBE|={stack_oof_score[0]:.2f}  "
             f"RMSE={stack_oof_score[1]:.2f}  score={stack_oof_score[2]:.2f}")

    # Apply to test
    # Need to align on the same ID order
    common_ids = test_cols[args.bases[0]].index
    X_test = np.column_stack([test_cols[t].reindex(common_ids).values for t in args.bases])
    test_pred = np.clip(ridge.predict(X_test), 0, 1361)

    sub = pd.DataFrame({"ID": common_ids, "TargetMBE": test_pred, "TargetRMSE": test_pred})
    sample = pd.read_csv(SAMP)
    final = sample[["ID"]].merge(sub, on="ID", how="left")
    out_path = SUBS / f"{args.out}.csv"
    final.to_csv(out_path, index=False)
    log.info(f"Saved: {out_path}  ({len(final):,} rows)")

    log_path = SUBS / f"{args.out}_log.txt"
    log_path.write_text(json.dumps({
        "bases":      args.bases,
        "alpha":      args.alpha,
        "intercept":  float(ridge.intercept_),
        "coefs":      {t: float(c) for t, c in zip(args.bases, ridge.coef_)},
        "oof_score":  {"mbe": stack_oof_score[0], "rmse": stack_oof_score[1],
                        "score": stack_oof_score[2]},
    }, indent=2))


if __name__ == "__main__":
    main()
