#!/usr/bin/env python3
"""Reversal benchmark files for the multi-objective EXAMM search (--objective mse_ir).

For each fold Y (validation year Y-1) and panel set, the 10-day z-score mean-reversion strategy traded
long-short 5/5 on the validation days, one daily return per EXAMM validation prediction:
  - signal on panel row r: -z_r, z_r = (P_r - mean(P_r-9..P_r)) / std(P_r-9..P_r) (sample std, ddof 1;
    undefined or zero std -> z = 0), P = split-adjusted price rebuilt from RET_raw over the whole panel,
    so the window uses only data up to row r (the same definition as core7_genome_frontier.py);
  - EXAMM (--time_offset 1) makes validation prediction j on val-file row j for val-file row j + 1, so
    benchmark day j = mean RET of the 5 stocks with the highest signal on row j minus mean RET of the 5
    lowest, RET taken from row j + 1 of the validation files themselves -- the same returns EXAMM uses
    for the genome's own long-short book, so both are in the same (scaled) units;
  - no trading costs (EXAMM has no prices); costs are applied later in the Mac backtest.
Writes benchmarks/panels_core7_examm_<Y>/<set>/reversal_ls55_val.csv (column reversal_ls55, one row per
validation prediction day) in this EXAMM-MO folder; nothing in EXAMM-Extended is changed.

Usage (from EXAMM-MO):
    python3 data_prep/write_core7_reversal_benchmark.py --years 2022 2023 2024
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

PANELS = "/anvil/projects/x-cis251123/shared/panels_core7"
EXAMM_DATA = "/anvil/scratch/x-dlopez6/experiments/EXAMM-Extended/datasets"
SETS = ["set1", "set2", "set3", "set4"]
WINDOW = 10
LONG_SHORT = 5


def reversal_signal(panel):
    stocks = [c[len("RET_raw_"):] for c in panel.columns if c.startswith("RET_raw_")]
    price = (1 + panel[[f"RET_raw_{t}" for t in stocks]].fillna(0)).cumprod()
    price.columns = stocks
    mean = price.rolling(WINDOW, min_periods=WINDOW).mean()
    std = price.rolling(WINDOW, min_periods=WINDOW).std(ddof=1)
    return -((price - mean) / std.where(std > 0)).fillna(0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024])
    ap.add_argument("--panels", default=PANELS)
    ap.add_argument("--data", default=EXAMM_DATA)
    ap.add_argument("--out_root", default="benchmarks")
    a = ap.parse_args()

    for y in a.years:
        for s in SETS:
            d = os.path.join(a.data, f"panels_core7_examm_{y}", s)
            info = pd.read_csv(os.path.join(d, "split_info.csv")).set_index("split")
            first, last = int(info.loc["val", "first_row"]), int(info.loc["val", "last_row"])
            stocks = sorted(f[:-len("_val.csv")] for f in os.listdir(d) if f.endswith("_val.csv"))
            panel = pd.read_csv(os.path.join(a.panels, f"{s}_core7", "panel_dates.csv")).set_index("row")
            missing = [t for t in stocks if f"RET_raw_{t}" not in panel.columns]
            if missing:
                sys.exit(f"{y} {s}: no RET_raw column for {missing}")
            sig = reversal_signal(panel).loc[first:last, stocks].to_numpy()
            ret = np.column_stack([pd.read_csv(os.path.join(d, f"{t}_val.csv"))["RET"].to_numpy() for t in stocks])
            if len(ret) != last - first + 1:
                sys.exit(f"{y} {s}: val files have {len(ret)} rows, split_info says {last - first + 1}")
            n_days = len(ret) - 1
            out = np.empty(n_days)
            for j in range(n_days):
                order = np.argsort(-sig[j], kind="stable")
                out[j] = ret[j + 1, order[:LONG_SHORT]].mean() - ret[j + 1, order[-LONG_SHORT:]].mean()
            o = os.path.join(a.out_root, f"panels_core7_examm_{y}", s)
            os.makedirs(o, exist_ok=True)
            pd.DataFrame({"reversal_ls55": out}).to_csv(os.path.join(o, "reversal_ls55_val.csv"), index=False)
            sharpe = out.mean() / out.std(ddof=1) * np.sqrt(252)
            print(f"{y} {s}: {n_days} validation days ({panel.loc[first + 1, 'date']} .. {panel.loc[last, 'date']}), "
                  f"{len(stocks)} stocks, benchmark Sharpe (gross, validation) {sharpe:+.2f}")


if __name__ == "__main__":
    main()
