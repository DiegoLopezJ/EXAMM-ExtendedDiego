#!/usr/bin/env python3
"""Checks EXAMM's in-search trading return (C++, rnn/trading_metric.cxx) against the Python backtest.

For each genome given, on its fold's validation year:
  1. runs build/rnn_examples/evaluate_trading_return, which prints the C++ net return and writes the raw forecasts it
     traded at full precision;
  2. trades exactly those forecasts with paper_long_short.py + stock.py (the backtest core7_main_results.py uses),
     on the prices / costs of trading_validation.csv -> must agree to rounding (~1e-12);
  3. trades the forecasts the Python pipeline itself uses (evaluate_rnn's evaluation_val files, written with 6
     significant digits) -> agrees up to that rounding; reported, not required to be exact.

Usage (from the EXAMM-Extended directory; needs the Financial_toolbox files, here analysis/):
    python3 analysis/check_cpp_trading_metric.py                       # every genome found in test_output (RET folds)
    python3 analysis/check_cpp_trading_metric.py --limit 5 --k 10
Exits with status 1 if a C++ return differs from the Python one on identical forecasts.
"""
import argparse
import contextlib
import glob
import io
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from Logger import Logger  # noqa: E402
from paper_long_short import paper_long_short_daily_values  # noqa: E402
from portfolio import Portfolio  # noqa: E402
from stock import Stock  # noqa: E402


def python_return(forecasts, trading, k):
    """Total net return of the paper's rule (paper_long_short.py, costs on) on forecasts[stock] -> array."""
    names = sorted(forecasts)
    logger = Logger()
    logger.set_level("ERROR")
    with contextlib.redirect_stdout(io.StringIO()):
        portfolio = Portfolio(names)
        portfolio.set_initial_spend_per_stock(100)
        portfolio.set_initial_capital(1000)
        portfolio.set_logger(logger)
        for n in names:
            c = Stock(n)
            c.return_prediction = np.asarray(forecasts[n], dtype=float)
            c.testing_period = len(c.return_prediction)
            c.stock_price = trading[f"PRC_{n}"].to_numpy(dtype=float)
            c.transaction_cost = trading[f"TC_{n}"].to_numpy(dtype=float)
            portfolio.add_company_to_protfolio(c)
            c.set_logger(logger)
        portfolio.set_use_TC(True)
        portfolio.set_trade_with_bid_ask(False)
        values, trades = paper_long_short_daily_values(portfolio, k, k, sign_check=True)
    daily = values[1:] / values[:-1] - 1
    return float(np.prod(1 + daily)) - 1, trades     # as core7_main_results.total_return (fraction)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examm", default=".", help="EXAMM-Extended directory")
    ap.add_argument("--binary", default="build/rnn_examples/evaluate_trading_return")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="check at most this many genomes (0: all)")
    ap.add_argument("--tolerance", type=float, default=1e-12)
    a = ap.parse_args()
    root = os.path.abspath(a.examm)
    binary = os.path.join(root, a.binary) if not os.path.isabs(a.binary) else a.binary

    # each repeat's FINAL global best (highest-numbered file), the genome evaluate_rnn's evaluation_val belongs to
    genomes = []
    for repeat_dir in sorted(glob.glob(os.path.join(root, "test_output", "core7_examm_ret_20[0-9][0-9]", "set[1-4]", "*"))):
        found = glob.glob(os.path.join(repeat_dir, "global_best_genome_*.bin"))
        if found:
            genomes.append(max(found, key=lambda f: int(f.rsplit("_", 1)[1][:-len(".bin")])))
    if a.limit:
        genomes = genomes[:a.limit]
    if not genomes:
        sys.exit("no genomes found under test_output/core7_examm_ret_<Y>/<set>/<repeat>/")

    worst, failures, rows = 0.0, 0, []
    for g in genomes:
        repeat_dir = os.path.dirname(g)
        s = os.path.basename(os.path.dirname(repeat_dir))
        year = os.path.basename(os.path.dirname(os.path.dirname(repeat_dir))).split("_")[-1]
        fold = os.path.join(root, "datasets", f"panels_core7_examm_{year}", s)
        trading_file = os.path.join(fold, "trading_validation.csv")
        if not os.path.exists(trading_file):
            print(f"skip {g}: no {trading_file} (run scripts/data_prep/write_core7_trading_files.py)")
            continue
        val_files = sorted(glob.glob(os.path.join(fold, "*_val.csv")))
        forecasts_file = os.path.join(repeat_dir, "cpp_trading_forecasts_val.csv")
        out = subprocess.run([binary, "--genome_file", g, "--testing_filenames", *val_files, "--time_offset", "1",
                              "--trading_file", trading_file, "--trading_k", str(a.k),
                              "--write_forecasts", forecasts_file], capture_output=True, text=True)
        if out.returncode != 0:
            print(f"FAILED to run on {g}:\n{out.stdout}{out.stderr}")
            failures += 1
            continue
        printed = dict(line.split(" ", 1) for line in out.stdout.strip().splitlines())
        cpp = float(printed["trading_return"])

        trading = pd.read_csv(trading_file)
        F = pd.read_csv(forecasts_file)
        os.remove(forecasts_file)
        py_same, trades = python_return({n: F[n].to_numpy() for n in F.columns}, trading, a.k)
        diff = abs(cpp - py_same)
        worst = max(worst, diff)
        ok = diff <= a.tolerance
        failures += not ok

        # the Python pipeline's own forecasts (evaluate_rnn's 6-digit files), if they exist
        py_pipeline = np.nan
        ev = sorted(glob.glob(os.path.join(repeat_dir, "evaluation_val", "*_val_predictions.csv")))
        if len(ev) == len(F.columns):
            med, iqr = trading["RET_MEDIAN"].iloc[0], trading["RET_IQR"].iloc[0]
            P = {}
            for f in ev:
                d = pd.read_csv(f)
                d.columns = [c.lstrip("#") for c in d.columns]
                P[os.path.basename(f)[:-len("_val_predictions.csv")]] = d["predicted_RET"].to_numpy() * iqr + med
            py_pipeline, _ = python_return(P, trading, a.k)

        rows.append({"genome": os.path.relpath(g, root), "cpp_%": cpp * 100, "python_same_forecasts_%": py_same * 100,
                     "abs_diff": diff, "python_pipeline_%": py_pipeline * 100, "days_traded": trades, "ok": ok})
        print(f"{'ok  ' if ok else 'FAIL'} {os.path.relpath(g, root)}: C++ {cpp * 100:+.6f} %  "
              f"Python {py_same * 100:+.6f} %  (diff {diff:.1e})  pipeline forecasts {py_pipeline * 100:+.6f} %  "
              f"traded {trades} days")

    print(f"\n{len(rows)} genome(s) checked, largest C++ vs Python difference {worst:.2e}, {failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
