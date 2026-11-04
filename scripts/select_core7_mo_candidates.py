#!/usr/bin/env python3
"""Candidate genomes of the multi-objective EXAMM runs, for cross-set evaluation and the genome frontier.

Each run (test_output/core7_examm_ret_mo_<Y>/<set>/<repeat>/) saved its Pareto archive in pareto_front.csv
(genome_id, val_mse, val_ir) with pareto_genome_<id>.bin. Archives of up to 50 genomes x 40 runs are too
many to evaluate on all 4 sets, so per fold and SOURCE SET the 10 repeats' archives are pooled and only the
genomes that no other genome of that set dominates (lower validation MSE and higher validation IR) are kept.
Pooling is per set because each set's MSE is in its own normalized units (EXAMM normalizes per set), so MSE
is not comparable across sets until the genomes are re-scored on all 4 sets (evaluation script, then the
frontier on the Mac in raw daily-return units). Each repeat's global best (lowest validation MSE, EXAMM's
own criterion) is added as well, so it can be compared with the multi-objective genomes.

Writes test_output/core7_examm_ret_mo_<Y>/candidates.csv: source_set, repeat, genome_id, file, val_mse,
val_ir, reason. Uses validation values only.
Usage (from the folder that holds test_output/, on Anvil or the Mac):
    python3 select_core7_mo_candidates.py --years 2022 2023 2024 [--root test_output]
"""
import argparse
import glob
import os
import re

import pandas as pd

SETS = ["set1", "set2", "set3", "set4"]
REPEATS = [str(r) for r in range(15, 25)]


def non_dominated(df):
    keep = []
    for i, r in df.iterrows():
        dominated = ((df["val_mse"] <= r["val_mse"]) & (df["val_ir"] >= r["val_ir"])
                     & ((df["val_mse"] < r["val_mse"]) | (df["val_ir"] > r["val_ir"]))).any()
        keep.append(not dominated)
    return df[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024])
    ap.add_argument("--root", default="test_output")
    ap.add_argument("--name", default="core7_examm_ret_mo", help="run folder prefix (core7_examm_ret_mo_test for the test run)")
    a = ap.parse_args()
    for y in a.years:
        base = os.path.join(a.root, f"{a.name}_{y}")
        rows, missing = [], []
        for s in SETS:
            pool, best = [], []
            for r in REPEATS:
                d = os.path.join(base, s, r)
                f = os.path.join(d, "pareto_front.csv")
                if not os.path.exists(f):
                    if os.path.isdir(d):
                        missing.append(f"{s}/{r}")
                    continue
                p = pd.read_csv(f)
                p.insert(0, "repeat", r)
                pool.append(p)
                gb = sorted(glob.glob(os.path.join(d, "global_best_genome_*.bin")),
                            key=lambda x: int(re.search(r"_(\d+)\.bin$", x).group(1)))
                if gb:
                    best.append({"repeat": r, "genome_id": int(re.search(r"_(\d+)\.bin$", gb[-1]).group(1)),
                                 "file": os.path.basename(gb[-1])})
            if not pool:
                continue
            pool = pd.concat(pool, ignore_index=True)
            front = non_dominated(pool).copy()
            front["file"] = [f"pareto_genome_{g}.bin" for g in front["genome_id"]]
            front["reason"] = "set Pareto front (val MSE vs val IR)"
            for b in best:
                hit = (front["repeat"] == b["repeat"]) & (front["genome_id"] == b["genome_id"])
                if hit.any():
                    front.loc[hit, "reason"] += "; repeat's lowest val MSE"
                else:
                    m = pool[(pool["repeat"] == b["repeat"]) & (pool["genome_id"] == b["genome_id"])]
                    front = pd.concat([front, pd.DataFrame([{
                        "repeat": b["repeat"], "genome_id": b["genome_id"], "file": b["file"],
                        "val_mse": m["val_mse"].iloc[0] if len(m) else float("nan"),
                        "val_ir": m["val_ir"].iloc[0] if len(m) else float("nan"),
                        "reason": "repeat's lowest val MSE (EXAMM criterion)"}])], ignore_index=True)
            front.insert(0, "source_set", s)
            rows.append(front[["source_set", "repeat", "genome_id", "file", "val_mse", "val_ir", "reason"]])
            print(f"{y} {s}: {len(pool)} archived genomes from {pool['repeat'].nunique()} runs -> "
                  f"{len(front)} candidates")
        if not rows:
            print(f"{y}: no pareto_front.csv under {base}")
            continue
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(os.path.join(base, "candidates.csv"), index=False)
        print(f"{y}: {len(out)} candidates -> {os.path.join(base, 'candidates.csv')}"
              + (f"  (runs without pareto_front.csv: {', '.join(missing)})" if missing else ""))


if __name__ == "__main__":
    main()
