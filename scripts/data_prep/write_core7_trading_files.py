#!/usr/bin/env python3
"""
Trading files for EXAMM's in-search trading-return metric (--fitness_metric return / --checkpoint_metric return).

For each test year Y and panel set, writes next to the fold's EXAMM files

    datasets/panels_core7_examm_<Y>/<set>/trading_validation.csv

(named so it does NOT match the "*_val.csv" pattern the job scripts pass to --validation_filenames)
with one row per row of the validation split (the rows of <STOCK>_val.csv, in the same order) and the columns

    row, date              panel row and date
    PRC_<STOCK>            the traded price, rebuilt from the unclipped raw return within the split
    TC_<STOCK>             the transaction cost per share (half-spread), scaled with that price
    RET_MEDIAN, RET_IQR    the panel set's robust-z statistics of RET (constant), to turn EXAMM's RET forecast
                           back into a raw return forecast: raw = RET * IQR + median

The prices and costs are built EXACTLY as core7_genome_frontier.py's price_dir() builds the Python backtest's price
files (prc = PRC_first * cumprod(1 + RET_raw) / first growth; tc = TC / PRC * prc), and written at full precision, so
the C++ metric trades on the same numbers as paper_long_short.py. Nothing else in the fold folder is changed; the
network's input files are untouched.

Usage (from the EXAMM-Extended directory):
    python3 scripts/data_prep/write_core7_trading_files.py --years 2022 2023 2024
    python3 scripts/data_prep/write_core7_trading_files.py --years 2024 --panels datasets/panels_core7
"""
import argparse
import json
import os
import sys

import pandas as pd

SETS = ["set1", "set2", "set3", "set4"]


def fail(msg):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(1)


def trading_frame(panel_dates, first, last, stats):
    """Prices and costs of rows first..last (inclusive), as core7_genome_frontier.price_dir() for the validation
    split (no warm-up rows to skip)."""
    rows = panel_dates.loc[first:last]
    stocks = [c[len("PRC_"):] for c in rows.columns if c.startswith("PRC_")]
    out = {"row": rows.index.to_numpy(), "date": rows["date"].to_numpy()}
    for s in stocks:
        growth = (1 + rows[f"RET_raw_{s}"]).cumprod()
        prc = rows[f"PRC_{s}"].iloc[0] * growth / growth.iloc[0]
        tc = rows[f"TC_{s}"] / rows[f"PRC_{s}"] * prc
        out[f"PRC_{s}"] = prc.to_numpy()
        out[f"TC_{s}"] = tc.to_numpy()
    frame = pd.DataFrame(out)
    frame["RET_MEDIAN"] = stats["median"]
    frame["RET_IQR"] = stats["iqr"]
    return frame, stocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024])
    ap.add_argument("--panels", default="datasets/panels_core7",
                    help="folder with <set>_core7/panel_dates.csv and panel_meta.json")
    ap.add_argument("--examm_root", default="datasets", help="folder with panels_core7_examm_<Y>/")
    a = ap.parse_args()

    for year in a.years:
        for s in SETS:
            fold = os.path.join(a.examm_root, f"panels_core7_examm_{year}", s)
            info_path = os.path.join(fold, "split_info.csv")
            if not os.path.exists(info_path):
                fail(f"{info_path} missing -- run split_core7_yearly.py first")
            info = pd.read_csv(info_path).set_index("split")
            first, last = int(info.loc["val", "first_row"]), int(info.loc["val", "last_row"])

            src = os.path.join(a.panels, f"{s}_core7")
            panel_dates = pd.read_csv(os.path.join(src, "panel_dates.csv")).set_index("row")
            stats = json.load(open(os.path.join(src, "panel_meta.json")))["stats"]["RET"]
            frame, stocks = trading_frame(panel_dates, first, last, stats)

            # the validation files EXAMM reads must exist for every stock and have one row per trading row
            for t in stocks:
                val_file = os.path.join(fold, f"{t}_val.csv")
                if not os.path.exists(val_file):
                    fail(f"{val_file} missing (stock {t} is in panel_dates.csv)")
                n = sum(1 for _ in open(val_file)) - 1
                if n != len(frame):
                    fail(f"{val_file} has {n} rows, the validation split has {len(frame)}")
            if frame.drop(columns=["date"]).isna().any().any():
                fail(f"{year} {s}: empty price or cost in the validation split")

            out = os.path.join(fold, "trading_validation.csv")
            frame.to_csv(out, index=False, lineterminator="\n")
            print(f"{year} {s}: {out} -- {len(stocks)} stocks, {len(frame)} rows "
                  f"({frame['date'].iloc[0]}..{frame['date'].iloc[-1]}), RET median {stats['median']}, IQR {stats['iqr']}")


if __name__ == "__main__":
    main()
