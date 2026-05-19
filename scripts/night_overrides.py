"""
Generate alternative nighttime-override submissions.

Takes a base submission (default: current best ensemble) and rewrites the
nighttime rows using three different strategies:

  1. force_zero          — night = 0 (physical hedge)
  2. per_station_median  — night = per-station MEDIAN (robust vs mean)
  3. per_station_hour    — night = per-(station, hour-of-day) mean,
                            computed from training rows with elev≤0

The daytime portion of the base submission is preserved unchanged. This
makes the comparison purely about night handling.

Outputs (next to the base submission):
  <BASE>__night_zero.csv
  <BASE>__night_median.csv
  <BASE>__night_per_hour.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
SUBS = ROOT / "submissions"
TRAIN_ENH = PROC / "Train_enhanced.csv"
TEST_ENH  = PROC / "Test_enhanced.csv"
TARGET = "radiation (W/m2)"


def build_override_tables():
    """Compute the three override tables from training data."""
    cols = ["station", "timestamp", "ext_sol_elevation", TARGET]
    train = pd.read_csv(TRAIN_ENH, usecols=cols, parse_dates=["timestamp"])
    night = train[train["ext_sol_elevation"] <= 0].copy()

    per_station_mean   = night.groupby("station")[TARGET].mean().to_dict()
    per_station_median = night.groupby("station")[TARGET].median().to_dict()

    night["hour"] = night["timestamp"].dt.hour
    per_station_hour = (night.groupby(["station", "hour"])[TARGET]
                        .mean()
                        .to_dict())

    return per_station_mean, per_station_median, per_station_hour


def apply_override(base_sub_path, test, night_mask, override_fn, out_path):
    sub = pd.read_csv(base_sub_path)
    # Align: sub may be in SampleSubmission order; test is in its own order.
    sub_lookup = sub.set_index("ID")
    test_ids = test["ID"].values

    new_pred = sub_lookup.loc[test_ids, "TargetMBE"].values.copy()
    night_indices = np.where(night_mask)[0]
    for i in night_indices:
        new_pred[i] = override_fn(test.iloc[i])
    new_pred = np.clip(new_pred, 0, 1361)

    # Reattach to base order (sample submission order)
    rebuilt = pd.DataFrame({"ID": test_ids, "TargetMBE": new_pred, "TargetRMSE": new_pred})
    final = sub[["ID"]].merge(rebuilt, on="ID", how="left")
    final.to_csv(out_path, index=False)
    print(f"  wrote {out_path.name}  "
          f"  night_overridden={night_mask.sum():,}  "
          f"  night_mean={new_pred[night_mask].mean():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(SUBS / "lgbm_ensemble_v1.csv"),
                    help="Base submission whose night rows we replace.")
    args = ap.parse_args()
    base_path = Path(args.base)
    tag = base_path.stem

    print(f"Base submission: {base_path}")
    print("Building override tables from training data ...")
    mean_t, median_t, hour_t = build_override_tables()
    fallback = float(np.mean(list(mean_t.values())))

    test = pd.read_csv(TEST_ENH,
                       usecols=["ID", "station", "timestamp", "ext_sol_elevation"],
                       parse_dates=["timestamp"])
    test["hour"] = test["timestamp"].dt.hour
    night_mask = (test["ext_sol_elevation"] <= 0).values
    print(f"Night rows in test: {night_mask.sum():,} / {len(test):,} "
          f"({night_mask.mean()*100:.1f}%)")

    print()
    print("Variant 1: force_zero")
    apply_override(base_path, test, night_mask,
                   lambda row: 0.0,
                   SUBS / f"{tag}__night_zero.csv")

    print("Variant 2: per_station_median")
    apply_override(base_path, test, night_mask,
                   lambda row: median_t.get(row["station"], fallback),
                   SUBS / f"{tag}__night_median.csv")

    print("Variant 3: per_station_hour")
    apply_override(base_path, test, night_mask,
                   lambda row: hour_t.get((row["station"], row["hour"]),
                                          mean_t.get(row["station"], fallback)),
                   SUBS / f"{tag}__night_per_hour.csv")

    print("\nQuick stats:")
    for name in ["night_zero", "night_median", "night_per_hour"]:
        p = SUBS / f"{tag}__{name}.csv"
        s = pd.read_csv(p)
        print(f"  {name:18s}  mean={s['TargetMBE'].mean():6.2f}  "
              f"median={s['TargetMBE'].median():6.2f}  "
              f"max={s['TargetMBE'].max():7.2f}")


if __name__ == "__main__":
    main()
