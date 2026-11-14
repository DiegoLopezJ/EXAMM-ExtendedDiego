#!/usr/bin/env python3
"""Tables for the Results section, split the way the professor asked, plus Mann-Whitney U tests.

  S1  retrain every year: test net return per test year and panel set, each genome choice vs the baseline
  S2  never retrain (fold-2022 runs kept for 2022-2024): the same
  S3  aggregated: over the 4 sets (per year), over the 3 years (per set), and everything together;
      plus the combined 4-set book from core7_main_results.py --pareto
  S4  the selection objectives: lowest validation MSE vs highest validation return vs the Pareto choice
      (win counts over the 12 set-years, the frontier sizes, validation vs test)
  TESTS  Mann-Whitney U (two-sided): two experiments A and B, each a distribution of test results of 10 repeated runs
      T1  retrain vs never retrain, per set and test year (10 vs 10 runs; test net return and test MSE),
          and pooled over the sets (40 vs 40)
      T2  K = 10 vs K = 5 book (same runs, different trading), if the 5/5 results exist
      T3  frontier runs vs runs off the frontier (is the validation frontier a better pool?)
      T4  Pareto choice vs baseline over the 12 set-years (12 vs 12; plus the paired Wilcoxon signed-rank test)

Genome choices per set and year: lowest val MSE (baseline, one run), Pareto choice (one run), highest val return
(one run), mean over the frontier runs, mean over all 10 runs (with its standard deviation). A run that never trades
in a year (net return exactly 0) has an undefined Sharpe, counted as 0; their number is shown per set and year.
"Net return" = total return over the test year after costs, in %.

Reads (written by core7_main_results.py --pareto, mean over the 10 seeds on by default):
  results/core7_main_pareto.csv, core7_main_repeats.csv, core7_main_benchmarks.csv, core7_main_pareto_summary.csv,
  core7_main_pareto_never_retrain.csv; for T2 also core7_main_k5_repeats.csv (core7_main_results.py --k 5 --pareto).
Writes results/core7_results_sections.xlsx (one sheet per subsection) and prints every table.
Run on the Mac:  cd ~/Financial_toolbox && python3 core7_results_sections.py && open results/core7_results_sections.xlsx
Options: --k 5 (use the 5/5 results as the main book; T2 then compares with 10/10).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

RESULTS = Path(__file__).resolve().parent / "results"
K = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv[:-1] else 10
TAG = "" if K == 10 else f"_k{K}"
OTHER_TAG = "_k5" if K == 10 else ""
YEARS = [2022, 2023, 2024]
SETS = ["set1", "set2", "set3", "set4"]
V = {s: f"V{i + 1}" for i, s in enumerate(SETS)}
RETRAIN, NEVER = "retrain every year", "never retrain (fold 2022 models)"
SINGLE = ["lowest val MSE", "Pareto choice", "highest val return"]


def need(name):
    p = RESULTS / name
    if not p.exists():
        sys.exit(f"missing {p} -- run: python3 core7_main_results.py --pareto" + ("" if K == 10 else f" --k {K}"))
    return pd.read_csv(p, dtype={"repeat": str})


pt = need(f"core7_main{TAG}_pareto.csv")
pt["choice"] = pt["choice"].fillna("")
rep = need(f"core7_main{TAG}_repeats.csv")
bench = pd.read_csv(RESULTS / f"core7_main{TAG}_benchmarks.csv") if (RESULTS / f"core7_main{TAG}_benchmarks.csv").exists() else None
pd.set_option("display.width", 220)
sheets = {}


def run_returns(design, year, s):
    """{repeat: (test net return %, test Sharpe, test MSE)} of the 10 runs used by a design in a test year and set."""
    g = rep[(rep["design"] == design) & (rep["test_year"] == year) & (rep["set"] == s)]
    # a run that never trades in the year (its forecasts never pass the sign gate) has net 0 and no volatility:
    # its Sharpe is undefined and is counted as 0 (no return, nothing at risk)
    return {r: (ret, 0.0 if not np.isfinite(sh) and abs(ret) < 1e-9 else sh, mse)
            for r, ret, sh, mse in zip(g["repeat"], g["total_return_%"], g["sharpe"], g["test_mse"])}


def choice_runs(fold, s):
    """The runs each choice uses, from the validation results of a fold (Pareto table)."""
    g = pt[(pt["test_year"] == fold) & (pt["set"] == s)]
    out = {c: g[g["choice"].str.contains(c, regex=False)]["repeat"].tolist() for c in SINGLE}
    out["mean over frontier runs"] = g[g["val_frontier"] == True]["repeat"].tolist()
    out["mean over all 10 runs"] = g["repeat"].tolist()
    return out


def per_set_table(design, fold_of):
    """Rows per test year and set: every choice's test net return and Sharpe, the 10-run spread, the benchmarks."""
    rows = []
    for y in YEARS:
        for s in SETS:
            runs = run_returns(design, y, s)
            if not runs:
                continue
            ch = choice_runs(fold_of(y), s)
            row = {"test year": y, "set": V[s]}
            for c, rs in ch.items():
                vals = [runs[r] for r in rs if r in runs]
                row[f"{c}: net %"] = float(np.mean([v[0] for v in vals])) if vals else np.nan
                row[f"{c}: Sharpe"] = float(np.mean([v[1] for v in vals])) if vals else np.nan
            allr = [v[0] for v in runs.values()]
            row["10 runs: sd of net %"] = float(np.std(allr, ddof=1))
            row["10 runs: min net %"], row["10 runs: max net %"] = float(min(allr)), float(max(allr))
            row["runs with positive net (of 10)"] = int(sum(v > 0 for v in allr))
            row["runs that never traded (of 10)"] = int(sum(abs(v) < 1e-9 for v in allr))
            row["frontier size"] = len(ch["mean over frontier runs"])
            row["Pareto minus baseline (net %)"] = row["Pareto choice: net %"] - row["lowest val MSE: net %"]
            if bench is not None:
                b = bench[(bench["test_year"] == y) & (bench["set"] == s)]
                if len(b):
                    row["buy-and-hold %"], row["equal weight %"] = float(b["bh_total_%"].iloc[0]), float(b["ew_total_%"].iloc[0])
            rows.append(row)
    return pd.DataFrame(rows)


def show(title, df, cols=None):
    print("\n" + title)
    print((df[cols] if cols else df).round(2).to_string(index=False))


NET_COLS = ["test year", "set", "lowest val MSE: net %", "Pareto choice: net %", "highest val return: net %",
            "mean over frontier runs: net %", "mean over all 10 runs: net %", "10 runs: sd of net %",
            "runs with positive net (of 10)", "Pareto minus baseline (net %)"]
print("=" * 100)
print(f"RESULTS SECTIONS (L/S {K}/{K}, costs on; net return = total return over the test year, %)")
print("=" * 100)
s1 = per_set_table(RETRAIN, lambda y: y)
s2 = per_set_table(NEVER, lambda y: YEARS[0])
show("S1  RETRAIN EVERY YEAR -- per test year and set:", s1, NET_COLS)
show("S2  NEVER RETRAIN (fold-2022 runs for 2022-2024) -- per test year and set:", s2, NET_COLS)
sheets["S1 retrain per set"] = s1
sheets["S2 never retrain per set"] = s2

# ---------------------------------------------------------------- S3 aggregates ----
num = [c for c in s1.columns if c.endswith(": net %") or c.endswith(": Sharpe")]
s3 = []
for design, df in ((RETRAIN, s1), (NEVER, s2)):
    for y, g in df.groupby("test year"):
        s3.append({"design": design, "level": "mean over the 4 sets", "test year": str(y), "set": "all", **g[num].mean().to_dict()})
    for st, g in df.groupby("set"):
        s3.append({"design": design, "level": "mean over the 3 years", "test year": "all", "set": st, **g[num].mean().to_dict()})
    s3.append({"design": design, "level": "mean over all 12 set-years", "test year": "all", "set": "all", **df[num].mean().to_dict()})
s3 = pd.DataFrame(s3)
show("S3  AGGREGATED (means of the per-set values above):", s3,
     ["design", "level", "test year", "set"] + [c for c in num if c.endswith("net %")])
book = []
for design, f in ((RETRAIN, f"core7_main{TAG}_pareto_summary.csv"), (NEVER, f"core7_main{TAG}_pareto_never_retrain.csv")):
    if (RESULTS / f).exists():
        b = pd.read_csv(RESULTS / f)
        b.insert(0, "design", design)
        book.append(b)
if book:
    book = pd.concat(book, ignore_index=True)
    piv = book.pivot_table(index=["design", "choice"], columns="period", values="net_return_%").round(2)
    pivs = book.pivot_table(index=["design", "choice"], columns="period", values="sharpe").round(2)
    print("\nS3  COMBINED 4-SET BOOK (core7_main_results.py --pareto), net return %:")
    print(piv.to_string())
    print("\n    Sharpe:")
    print(pivs.to_string())
    order = ["lowest val MSE", "Pareto choice", "highest val return", "average over the frontier runs",
             "average over the 10 runs"]
    periods = [str(y) for y in YEARS] + [f"{YEARS[0]}-{YEARS[-1]}"]
    tidy = []
    for design in book["design"].unique():
        for ch in [c for c in order if c in set(book["choice"])]:
            g = book[(book["design"] == design) & (book["choice"] == ch)].set_index("period")
            row = {"design": design, "choice": ch}
            for per in periods:
                row[f"net % {per}"] = g["net_return_%"].get(per)
            for per in periods:
                row[f"Sharpe {per}"] = g["sharpe"].get(per)
            row[f"max drawdown % {periods[-1]}"] = g["max_dd_%"].get(periods[-1])
            tidy.append(row)
    sheets["S3 combined book"] = pd.DataFrame(tidy)
sheets["S3 aggregated"] = s3

# ---------------------------------------------------------------- S3 per panel, 2022-2024 joined ----
# Sharpe and max drawdown of each panel over its JOINED 2022-2024 daily returns (the definition of the per-set Table 8),
# for every choice. Single choices: the chosen run's daily returns joined over the three years (with retraining a
# different run each year, as traded). Averages: the joined metrics of each run (10-run average: run r in every year)
# or of each combination of one frontier run per year (frontier average), then averaged.
import itertools
CACHE = RESULTS / "genome_frontier" / f"stale_trades_paper{'' if K == 5 else f'_k{K}'}.npz"


def joined_metrics(r):
    r = np.asarray(r, dtype=float)
    value = np.cumprod(1 + r)
    vol = r.std()
    sharpe = r.mean() / vol * np.sqrt(252) if vol > 1e-12 else 0.0       # a book that never trades: 0
    return sharpe, -float((1 - value / np.maximum.accumulate(value)).max()) * 100


if CACHE.exists():
    cache = np.load(CACHE)
    pp, pmiss = [], []
    for design, fold_of in ((RETRAIN, lambda y: y), (NEVER, lambda y: YEARS[0])):
        for s in SETS:
            per_year = {y: choice_runs(fold_of(y), s) for y in YEARS}
            def series(path):                       # path: one run per year -> joined daily returns, or None
                keys = [f"{fold_of(y)}|{s}|{r}|{y}" for y, r in zip(YEARS, path)]
                if any(k not in cache for k in keys):
                    pmiss.append(f"{design} {V[s]}")
                    return None
                return np.concatenate([np.asarray(cache[k], float) for k in keys])
            for choice in SINGLE + ["mean over frontier runs", "mean over all 10 runs"]:
                if choice in SINGLE:
                    paths = [[per_year[y][choice][0] if per_year[y][choice] else None for y in YEARS]]
                elif choice == "mean over all 10 runs":
                    paths = [[r] * len(YEARS) for r in per_year[YEARS[0]][choice]]
                else:
                    paths = [list(c) for c in itertools.product(*[per_year[y][choice] for y in YEARS])]
                ms = []
                for path in paths:
                    if None in path:
                        continue
                    r = series(path)
                    if r is not None:
                        ms.append(joined_metrics(r))
                if not ms:
                    continue
                df = s1 if design == RETRAIN else s2
                net = df[df["set"] == V[s]][f"{choice}: net %"].mean()
                pp.append({"design": design, "choice": choice, "panel": V[s], "net % (mean of years)": float(net),
                           "Sharpe 2022-2024": float(np.mean([m[0] for m in ms])),
                           "max drawdown % 2022-2024": float(np.mean([m[1] for m in ms])), "books averaged": len(ms)})
    if pmiss:
        print(f"\n!! per-panel metrics: some runs not traded yet ({len(set(pmiss))} design/panel cases), e.g. "
              f"{sorted(set(pmiss))[:3]} -- run core7_main_results.py --pareto" + ("" if K == 10 else f" --k {K}"))
    if pp:
        pp = pd.DataFrame(pp)
        avg = pp.groupby(["design", "choice"], sort=False)[["net % (mean of years)", "Sharpe 2022-2024",
                                                            "max drawdown % 2022-2024"]].mean().reset_index()
        avg.insert(2, "panel", "Average V1-V4")
        avg["books averaged"] = pp.groupby(["design", "choice"], sort=False)["books averaged"].sum().to_numpy()
        pp = pd.concat([pp, avg], ignore_index=True)
        order = {c: i for i, c in enumerate(SINGLE + ["mean over frontier runs", "mean over all 10 runs"])}
        dorder = {RETRAIN: 0, NEVER: 1}
        pp = pp.sort_values(["design", "choice", "panel"],
                            key=lambda c: c.map(order) if c.name == "choice" else c.map(dorder) if c.name == "design" else c,
                            kind="stable").reset_index(drop=True)
        pp["books averaged"] = pp["books averaged"].astype(int)
        print("\nS3  PER PANEL, 2022-2024 JOINED (Sharpe and max drawdown over each panel's joined daily returns):")
        show_cols = ["design", "choice", "panel", "net % (mean of years)", "Sharpe 2022-2024", "max drawdown % 2022-2024"]
        print(pp[pp["panel"] == "Average V1-V4"][show_cols].round(2).to_string(index=False))
        sheets["S3 per panel 2022-2024"] = pp
else:
    print(f"\n(per-panel Sharpe / drawdown skipped: no {CACHE.name}; run core7_main_results.py --pareto first)")

# ---------------------------------------------------------------- S4 the selection objectives ----
s4 = []
for design, df in ((RETRAIN, s1), (NEVER, s2)):
    for a, b in (("Pareto choice", "lowest val MSE"), ("Pareto choice", "highest val return"),
                 ("highest val return", "lowest val MSE"), ("Pareto choice", "mean over all 10 runs"),
                 ("lowest val MSE", "mean over all 10 runs")):
        d = df[f"{a}: net %"] - df[f"{b}: net %"]
        s4.append({"design": design, "A": a, "B": b, "set-years where A > B": f"{int((d > 0).sum())} of {int(d.notna().sum())}",
                   "mean A - B (net %)": float(d.mean()), "median A - B (net %)": float(d.median())})
s4 = pd.DataFrame(s4)
show("S4  SELECTION OBJECTIVES -- head to head over the 12 set-years (net return):", s4)
vt = pt.copy()
for c in SINGLE:
    vt[c] = vt["choice"].str.contains(c, regex=False)
obj = []
for c in SINGLE:
    g = vt[vt[c]]
    obj.append({"choice": c, "val MSE (mean)": g["val_mse"].mean(), "val net % (mean)": g["val_return_%"].mean(),
                "test net % (mean)": g["test_return_%"].mean(), "test MSE (mean)": g["test_mse"].mean()})
obj = pd.DataFrame(obj)
print("\n    Validation vs test of the chosen runs (retrain every year, mean over the 12 set-years):")
print(obj.to_string(index=False, float_format=lambda v: f"{v:.6f}" if abs(v) < 0.01 else f"{v:.2f}"))
sheets["S4 objectives"] = s4
sheets["S4 validation vs test"] = obj


# ---------------------------------------------------------------- Mann-Whitney U tests ----
def mwu(a, b, label, metric, A, B):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return None
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return {"test": label, "metric": metric, "A": A, "B": B, "n A": len(a), "n B": len(b),
            "median A": float(np.median(a)), "median B": float(np.median(b)), "U": float(u), "p": float(p),
            "significant (p < 0.05)": "yes" if p < 0.05 else "no"}


tests = []
for metric, col in (("test net return %", "total_return_%"), ("test MSE", "test_mse"), ("test Pearson IC", "test_ic_pearson")):
    for y in YEARS[1:]:                                  # 2022: both designs use the same runs
        A_all, B_all = [], []
        for s in SETS:
            a = rep[(rep["design"] == RETRAIN) & (rep["test_year"] == y) & (rep["set"] == s)][col]
            b = rep[(rep["design"] == NEVER) & (rep["test_year"] == y) & (rep["set"] == s)][col]
            A_all += a.tolist()
            B_all += b.tolist()
            t = mwu(a, b, f"T1 retrain vs never retrain, {y} {V[s]}", metric, "retrain (10 runs)", "never retrain (10 runs)")
            if t:
                tests.append(t)
        t = mwu(A_all, B_all, f"T1 retrain vs never retrain, {y} all sets", metric, "retrain (40 runs)", "never retrain (40 runs)")
        if t:
            tests.append(t)
other = RESULTS / f"core7_main{OTHER_TAG}_repeats.csv"
if other.exists():
    rep2 = pd.read_csv(other, dtype={"repeat": str})
    k2 = 5 if K == 10 else 10
    for design in (RETRAIN, NEVER):
        for y in YEARS:
            a = rep[(rep["design"] == design) & (rep["test_year"] == y)]["total_return_%"]
            b = rep2[(rep2["design"] == design) & (rep2["test_year"] == y)]["total_return_%"]
            t = mwu(a, b, f"T2 K={K} vs K={k2}, {design}, {y} (all sets)", "test net return %", f"K={K} (40 runs)", f"K={k2} (40 runs)")
            if t:
                tests.append(t)
else:
    print(f"\n(T2 skipped: no {other.name}; run core7_main_results.py --k {5 if K == 10 else 10} --pareto first)")
for y in YEARS:
    g = pt[pt["test_year"] == y]
    t = mwu(g[g["val_frontier"] == True]["test_return_%"], g[g["val_frontier"] != True]["test_return_%"],
            f"T3 frontier vs off-frontier runs, {y} (all sets, retrain)", "test net return %", "frontier runs", "other runs")
    if t:
        tests.append(t)
t = mwu(pt[pt["val_frontier"] == True]["test_return_%"], pt[pt["val_frontier"] != True]["test_return_%"],
        "T3 frontier vs off-frontier runs, 2022-2024 (all, retrain)", "test net return %", "frontier runs", "other runs")
if t:
    tests.append(t)
for design, df in ((RETRAIN, s1), (NEVER, s2)):
    a, b = df["Pareto choice: net %"], df["lowest val MSE: net %"]
    t = mwu(a, b, f"T4 Pareto vs baseline, {design}", "test net return %", "Pareto (12 set-years)", "baseline (12 set-years)")
    if t:
        d = (a - b).dropna()
        with warnings.catch_warnings():          # set-years where both choose the same run give 0 differences
            warnings.simplefilter("ignore")
            t["paired Wilcoxon p"] = float(wilcoxon(d).pvalue) if (d != 0).any() else np.nan
        tests.append(t)
tests = pd.DataFrame(tests)
paired = pd.DataFrame()
if "paired Wilcoxon p" in tests:
    paired = tests[tests["paired Wilcoxon p"].notna()][["test", "metric", "A", "B", "n A", "median A", "median B",
                                                          "paired Wilcoxon p"]].copy()
    paired.insert(5, "pairs", paired.pop("n A"))
    paired["significant (p < 0.05)"] = np.where(paired["paired Wilcoxon p"] < 0.05, "yes", "no")
    tests = tests.drop(columns="paired Wilcoxon p")
print("\n" + "=" * 100)
print("MANN-WHITNEY U TESTS (two-sided; p < 0.05 = the two distributions differ significantly)")
print("=" * 100)
print(tests.round(4).to_string(index=False))
sheets["Mann-Whitney tests"] = tests
if len(paired):
    print("\nPAIRED WILCOXON SIGNED-RANK (Pareto vs baseline, the same 12 set-years paired):")
    print(paired.round(4).to_string(index=False))
    sheets["Wilcoxon paired (T4)"] = paired

# ---------------------------------------------------------------- workbook ----
try:
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    out = RESULTS / f"core7_results_sections{TAG}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            for c in ws[1]:
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = PatternFill("solid", fgColor="1F3864")
                c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
            ws.row_dimensions[1].height = 48
            for i, col in enumerate(df.columns, 1):
                ws.column_dimensions[get_column_letter(i)].width = max(10, min(40, max(len(str(w)) for w in str(col).split()) + 4,
                                                                               int(df[col].astype(str).str.len().max() or 0) + 2))
                for row in ws.iter_rows(min_row=2, min_col=i, max_col=i):
                    for c in row:
                        if isinstance(c.value, float):
                            col_s = str(col)
                            metric = str(ws.cell(row=c.row, column=list(df.columns).index("metric") + 1).value) if "metric" in df.columns else ""
                            mse_col = col_s.startswith(("val MSE", "test MSE")) or (col_s.startswith("median") and metric == "test MSE")
                            ic_col = col_s.startswith("median") and "IC" in metric
                            if mse_col:
                                c.number_format = "0.000000"
                            elif col_s in ("p", "paired Wilcoxon p") or ic_col:
                                c.number_format = "0.0000"
                            else:
                                c.number_format = "0.00"
                        if col == "significant (p < 0.05)" and c.value == "yes":
                            c.fill = PatternFill("solid", fgColor="E2EFDA")
            ws.freeze_panes = "C2"
    print(f"\nSaved {out}")
except ImportError:
    print("\nopenpyxl not installed -- no workbook (python3 -m pip install --user openpyxl)")
