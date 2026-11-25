#!/usr/bin/env python3
"""PARETO SEARCH, step 3: compare the Pareto EXAMM search with core7-ret (PARETO_SEARCH_PROTOCOL.md section 4).

Reads CSVs only (nothing is re-traded):
  results/core7_main_repeats.csv           core7-ret, every seed x test year (core7_main_results.py, run with the
                                           corrected stock.py); design "retrain every year"
  results/core7_pareto_search_repeats.csv  the Pareto search, every run x test year: the Pareto choice and both
                                           front ends (core7_pareto_search_results.py)

Tables (test years; every number is a mean over the 10 runs / seeds unless stated):
  T1  per panel and test year: test net return %, core7-ret vs Pareto choice vs the two front ends
  T2  aggregated: per test year (over the 4 panels), per panel (over the 3 years), and overall; return, Sharpe,
      max drawdown, Rank IC, Pearson IC, test MSE
  T3  tests: per panel and year, two-sided Mann-Whitney U, 10 core7-ret seeds vs 10 Pareto-choice runs (test net
      return); over the 12 panel-years, the paired Wilcoxon signed-rank test on the per-cell means
  T4  validation vs test for the Pareto search (the search used the validation year, so validation is optimistic)
Writes results/core7_pareto_search_vs_ret.xlsx (one sheet per table) and the same tables as CSVs.

Run on the Mac:   cd ~/Financial_toolbox && python3 core7_pareto_search_compare.py
                  python3 core7_pareto_search_compare.py --run core7_examm_ret_pareto_ckret
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

RESULTS = Path(__file__).resolve().parent / "results"
RUN = "core7_examm_ret_pareto"
if "--run" in sys.argv[:-1]:
    RUN = sys.argv[sys.argv.index("--run") + 1]
OUT = RUN.replace("core7_examm_ret_pareto", "core7_pareto_search")
BASELINE = "core7-ret (mean of seeds)"
CHOICES = ["Pareto choice", "lowest val MSE end", "highest val return end"]
METRICS = ["total_return_%", "sharpe", "max_dd_%", "test_rank_ic", "test_ic_pearson", "test_mse"]


def load():
    base_path, new_path = RESULTS / "core7_main_repeats.csv", RESULTS / f"{OUT}_repeats.csv"
    for p, how in ((base_path, "python3 core7_main_results.py --pareto"), (new_path, "python3 core7_pareto_search_results.py")):
        if not p.exists():
            sys.exit(f"missing {p} -- run: {how}")
    base = pd.read_csv(base_path)
    base = base[base["design"] == "retrain every year"].copy()
    base["experiment"] = BASELINE
    new = pd.read_csv(new_path)
    new["experiment"] = "Pareto search: " + new["choice"]
    cols = ["experiment", "test_year", "set", "panel", "repeat"] + METRICS
    both = pd.concat([base[cols], new[cols]], ignore_index=True)
    return base, new, both


def per_cell(both):
    t = both.pivot_table(index=["test_year", "panel"], columns="experiment", values="total_return_%", aggfunc="mean")
    order = [BASELINE] + [f"Pareto search: {c}" for c in CHOICES]
    t = t[[c for c in order if c in t.columns]]
    t["Pareto choice - core7-ret"] = t["Pareto search: Pareto choice"] - t[BASELINE]
    return t.reset_index()


def aggregated(both):
    parts = []
    for level, keys in (("per test year", ["test_year"]), ("per panel", ["panel"]), ("overall", [])):
        g = both.groupby(["experiment"] + keys)[METRICS].mean().reset_index()
        g.insert(0, "level", level)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def tests(base, new):
    pick = new[new["choice"] == "Pareto choice"]
    rows = []
    for (y, panel), b in base.groupby(["test_year", "panel"]):
        p = pick[(pick["test_year"] == y) & (pick["panel"] == panel)]
        if len(b) < 2 or len(p) < 2:
            continue
        u = mannwhitneyu(b["total_return_%"], p["total_return_%"], alternative="two-sided")
        rows.append({"test": "Mann-Whitney U (two-sided), 10 vs 10 runs", "test_year": y, "panel": panel,
                     "n_core7_ret": len(b), "n_pareto": len(p), "mean_core7_ret_%": b["total_return_%"].mean(),
                     "mean_pareto_%": p["total_return_%"].mean(), "statistic": u.statistic, "p_value": u.pvalue})
    T = pd.DataFrame(rows)
    if len(T) >= 2:
        diff = T["mean_pareto_%"] - T["mean_core7_ret_%"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = wilcoxon(T["mean_pareto_%"], T["mean_core7_ret_%"]) if (diff != 0).any() else None
        T = pd.concat([T, pd.DataFrame([{
            "test": f"paired Wilcoxon signed-rank over the {len(T)} panel-years (per-cell means)",
            "mean_core7_ret_%": T["mean_core7_ret_%"].mean(), "mean_pareto_%": T["mean_pareto_%"].mean(),
            "statistic": np.nan if w is None else w.statistic, "p_value": np.nan if w is None else w.pvalue,
            "panel": f"Pareto better in {(diff > 0).sum()} of {len(T)}"}])], ignore_index=True)
    return T


def val_vs_test(new):
    g = new.groupby(["choice", "test_year"])[["val_return_%", "total_return_%"]].mean().reset_index()
    g = g.rename(columns={"total_return_%": "test_return_%"})
    g["test - validation (pp)"] = g["test_return_%"] - g["val_return_%"]
    return g


def main():
    base, new, both = load()
    T1, T2, T3, T4 = per_cell(both), aggregated(both), tests(base, new), val_vs_test(new)
    pd.set_option("display.width", 200)
    print("T1  test net return % per panel and year (mean over runs / seeds)\n" + T1.round(2).to_string(index=False))
    print("\nT2  overall (mean over all panels, years and runs)\n"
          + T2[T2["level"] == "overall"].drop(columns="level").dropna(axis=1, how="all").round(4).to_string(index=False))
    print("\nT3  tests (test net return)\n" + T3.round(4).to_string(index=False))
    print("\nT4  validation vs test, Pareto search\n" + T4.round(2).to_string(index=False))

    sheets = {"T1 per panel-year": T1, "T2 aggregated": T2, "T3 tests": T3, "T4 val vs test": T4}
    for name, t in sheets.items():
        t.to_csv(RESULTS / f"{OUT}_vs_ret_{name.split()[0]}.csv", index=False)
    xlsx = RESULTS / f"{OUT}_vs_ret.xlsx"
    try:
        with pd.ExcelWriter(xlsx) as writer:
            notes = pd.DataFrame({"note": [
                "core7-ret: core7_main_repeats.csv, retrain every year, mean over the 10 seeds (corrected stock.py).",
                f"Pareto search: {RUN}, chosen per run by PARETO_SEARCH_PROTOCOL.md (validation only), mean over runs.",
                "Test years only are evidence; the search used the validation year to keep networks that trade well.",
                "Return = total net return over the test year in %, paper's rule (K = 10, sign check, costs on)."]})
            notes.to_excel(writer, sheet_name="About", index=False)
            for name, t in sheets.items():
                t.to_excel(writer, sheet_name=name, index=False)
        print(f"\nSaved {xlsx} and the tables as CSVs ({OUT}_vs_ret_T*.csv)")
    except ImportError:
        print(f"\n(openpyxl is not installed: only the CSVs were written, {OUT}_vs_ret_T*.csv)")


if __name__ == "__main__":
    main()
