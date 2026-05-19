"""
Ensemble multiple submission CSVs by simple averaging.

Usage:
  python scripts/ensemble.py file1.csv file2.csv [file3.csv ...] --out submissions/ensemble.csv

The Zindi format requires TargetMBE == TargetRMSE per row, so we
average just one column and replicate it.
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Submission CSVs to average")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="Optional weights matching inputs (will be normalised)")
    args = ap.parse_args()

    if args.weights:
        if len(args.weights) != len(args.inputs):
            ap.error(f"weights count {len(args.weights)} != inputs count {len(args.inputs)}")
        w = np.array(args.weights, dtype=float)
        w = w / w.sum()
    else:
        w = np.ones(len(args.inputs)) / len(args.inputs)

    base = pd.read_csv(args.inputs[0])[["ID"]]
    accum = np.zeros(len(base), dtype=float)
    for path, weight in zip(args.inputs, w):
        sub = pd.read_csv(path).set_index("ID").loc[base["ID"]].reset_index()
        # Use TargetMBE (== TargetRMSE) as the per-row prediction
        accum += weight * sub["TargetMBE"].values
        print(f"  loaded {path}  weight={weight:.3f}  "
              f"mean={sub['TargetMBE'].mean():.2f}  rows={len(sub):,}")

    accum = np.clip(accum, 0, 1361)
    out = pd.DataFrame({"ID": base["ID"], "TargetMBE": accum, "TargetRMSE": accum})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nWrote ensemble: {out_path}  ({len(out):,} rows, "
          f"mean={accum.mean():.2f}, min={accum.min():.2f}, max={accum.max():.2f})")


if __name__ == "__main__":
    main()
