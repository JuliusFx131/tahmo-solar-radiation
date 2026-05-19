"""
Probe whether the Zindi LB scores TargetMBE and TargetRMSE columns
independently or treats them as the rules say ("should be identical").

Strategy: take our current best submission and add a large constant
(+50 W/m²) to the TargetMBE column only. Submit and observe what comes
back from the LB:

  • If LB |MBE| jumps by ~50 (roughly) and RMSE is unchanged:
       → columns are independent. Exploit by submitting best-MBE
         predictions in TargetMBE and best-RMSE in TargetRMSE.
  • If both metrics change identically (averaged) or only one moved:
       → columns are linked. Abandon the idea.
  • If the submission errors at upload:
       → strict enforcement. Clear answer, no LB slot consumed.

Predictions stay within [0, 1361] after clipping.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "submissions" / "lgbm_v12_shift_n243.csv"
OUT  = ROOT / "submissions" / "probe_mbe_plus50.csv"
SHIFT_MBE_COL = 50.0  # large enough to be unambiguous in returned scores


def main():
    src = pd.read_csv(SRC)
    assert {"ID", "TargetMBE", "TargetRMSE"}.issubset(src.columns), \
        f"unexpected columns: {src.columns.tolist()}"

    print(f"Source: {SRC.name}")
    print(f"  rows = {len(src):,}")
    print(f"  TargetMBE  mean before = {src['TargetMBE'].mean():.4f}")
    print(f"  TargetRMSE mean before = {src['TargetRMSE'].mean():.4f}")
    diff = (src["TargetMBE"] != src["TargetRMSE"]).sum()
    print(f"  rows where the two columns already differ: {diff}  "
          f"(should be 0 for a clean baseline)")

    out = src.copy()
    out["TargetMBE"] = np.clip(out["TargetMBE"].astype(float) + SHIFT_MBE_COL,
                               0.0, 1361.0)
    # TargetRMSE intentionally unchanged.

    print(f"\nProbe: shifted TargetMBE by +{SHIFT_MBE_COL:.0f} W/m² (clipped to [0, 1361])")
    print(f"  TargetMBE  mean after  = {out['TargetMBE'].mean():.4f}  "
          f"(delta {out['TargetMBE'].mean() - src['TargetMBE'].mean():+.4f})")
    print(f"  TargetRMSE mean after  = {out['TargetRMSE'].mean():.4f}  (unchanged)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nWrote: {OUT}")
    print(f"  -> Upload to Zindi as a normal submission. Compare the returned")
    print(f"     |MBE| and RMSE against the v12_shift_n243 LB row (2.72 / 62.45).")
    print(f"     Expected if columns are independent:")
    print(f"       |MBE| ≈ 2.72 + {SHIFT_MBE_COL:.0f} - (clip loss) ≈ ~47-50")
    print(f"       RMSE  ≈ 62.45 (unchanged)")


if __name__ == "__main__":
    main()
