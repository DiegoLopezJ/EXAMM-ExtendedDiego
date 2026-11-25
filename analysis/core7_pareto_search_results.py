#!/usr/bin/env python3
"""PARETO SEARCH, step 2: check, choose and backtest the genomes of the multi-objective EXAMM runs.

Implements PARETO_SEARCH_PROTOCOL.md exactly. The runs (scripts/base_run/core7_examm_yearly_ret_pareto.sh) kept
genomes by Pareto rank on validation MSE and validation return; each saved its front in pareto_front.csv. Here:

  1. VALIDATION CHECK: every front genome's validation return is recomputed in Python from its evaluate_rnn
     forecasts (paper_long_short.py, K = 10, sign check, costs on) and compared with the val_return EXAMM wrote.
     They agree to the forecast files' rounding (6 significant digits); the script stops if they do not, since the
     search and the backtest would then disagree.
  2. CHOICE (validation only): per run, among its own front, the lowest rank sum of validation MSE (ascending) and
     validation return (descending), both ranked over the fronts of ALL runs of that panel and fold (as the
     after-the-fact --pareto choice ranks over the 10 repeats; ranked within one front the sum is the same for every
     genome); ties -> lower validation MSE. The run's two front ends are recorded too.
  3. TEST: retrain every year -- each fold's chosen genomes traded on that fold's test year with the paper's rule
     (K = 10, sign check, the corrected stock.py costs), capital 1000; total net return, Sharpe, max drawdown,
     test MSE (raw daily-return units), Rank IC and Pearson IC, per run -- the columns of core7_main_repeats.csv.

Needs, in EXAMM_DIR (default ~/EXAMM-Extended): test_output/<RUN>_<Y>/<set>/<repeat>/pareto_front.csv and the
forecasts of scripts/rnn_examples/evaluate_rnn_core7_pareto.sh (evaluation_g<id>_val/, evaluation_g<id>_test/);
datasets/panels_core7_examm_<Y>/<set>/split_info.csv and the core7 panels.

Run on the Mac:   cd ~/Financial_toolbox && python3 core7_pareto_search_results.py
                  python3 core7_pareto_search_results.py --run core7_examm_ret_pareto_ckret   (CHECKPOINT=return runs)
Writes results/core7_pareto_search[_ckret]_genomes.csv (every front genome, validation and test),
       results/core7_pareto_search[_ckret]_repeats.csv (per run and test year: the Pareto choice and both front ends),
and prints the mean over the 10 runs per set and year. Trades are cached in
results/genome_frontier/pareto_search_trades.npz (delete it if forecasts or stock.py change).
Then: python3 core7_pareto_search_compare.py
"""
import contextlib
import io
import sys

import numpy as np
import pandas as pd

from core7_genome_frontier import (EXAMM, REPEATS, RESULTS, SETS, WORK, YEARS, forecast_scores, metrics, price_dir,
                                   prediction_dir, read_predictions, ret_stats, scored_mask)

K = 10                        # the paper's book: long 10 / short 10 (PARETO_SEARCH_PROTOCOL.md)
RUN = "core7_examm_ret_pareto"
if "--run" in sys.argv[:-1]:
    RUN = sys.argv[sys.argv.index("--run") + 1]
OUT = RUN.replace("core7_examm_ret_pareto", "core7_pareto_search")
RUN_TAG = "" if RUN == "core7_examm_ret_pareto" else RUN.replace("core7_examm_ret_pareto", "")
CHECK_TOLERANCE = 1e-4        # validation return, as a fraction (0.01 percentage points)
CHOICES = ["Pareto choice", "lowest val MSE end", "highest val return end"]


def paper_returns(pred_d, price_d):
    """Daily returns of the paper's Algorithm 1 on one set (paper_long_short.py), costs on.
    The same function as core7_main_results.py (which cannot be imported: it runs on import)."""
    from Logger import Logger
    from paper_long_short import paper_long_short_daily_values
    from portfolio import Portfolio
    from stock import Stock
    names = sorted(f.name[:-len("_predictions.csv")] for f in pred_d.glob("*_predictions.csv"))
    logger = Logger()
    logger.set_level("ERROR")
    with contextlib.redirect_stdout(io.StringIO()):
        portfolio = Portfolio(names)
        portfolio.set_initial_spend_per_stock(100)
        portfolio.set_initial_capital(1000)
        portfolio.set_logger(logger)
        for n in names:
            c = Stock(n)
            c.read_return_prediction(str(pred_d / f"{n}_predictions.csv"))
            c.read_stock_price(str(price_d / f"{n}.csv"))
            portfolio.add_company_to_protfolio(c)
            c.set_logger(logger)
        portfolio.set_use_TC(True)
        portfolio.set_trade_with_bid_ask(False)
        values, trades = paper_long_short_daily_values(portfolio, K, K, sign_check=True)
    return values[1:] / values[:-1] - 1, trades


def total_return(r):
    """Total return over the period in % ((C - C0) / C0, the paper's net return), as core7_main_results.py."""
    return (float(np.prod(1 + np.asarray(r, dtype=float))) - 1) * 100


def pearson_ic(P, year, s, split):
    """Mean daily cross-sectional PEARSON correlation of the raw forecast with the raw next-day return."""
    st = ret_stats(s)
    rows, keep = scored_mask(year, s, split, len(P))
    raw = (P * st["iqr"] + st["median"])[keep].reset_index(drop=True)
    actual = pd.DataFrame({t: rows[f"RET_raw_{t}"].to_numpy() for t in P.columns})[keep].reset_index(drop=True)
    return float(raw.corrwith(actual, axis=1).dropna().mean())


def choose(front, pooled):
    """PARETO_SEARCH_PROTOCOL.md section 2, with EXAMM's own validation numbers. front: one run's pareto_front.csv;
    pooled: the fronts of all runs of the same panel and fold (including this one). Returns the genome ids of the
    Pareto choice, the run's lowest-validation-MSE end and its highest-validation-return end.

    The ranks are taken over the pooled fronts: within ONE front, the genomes ordered by MSE are ordered by return in
    reverse, so the rank sum is the same for all of them and the choice would always be the lowest-MSE end."""
    rank_mse = pooled["val_mse"].rank(method="min")
    rank_ret = pooled["val_return"].rank(method="min", ascending=False)
    score = (rank_mse + rank_ret).to_numpy()
    own = front.copy()
    own["score"] = [score[(pooled["run"] == r).to_numpy() & (pooled["genome_id"] == g).to_numpy()][0]
                    for r, g in zip(own["run"], own["genome_id"])]
    tied = own[own["score"] == own["score"].min()]
    pick = tied.loc[tied["val_mse"].idxmin(), "genome_id"]
    low_mse = front.loc[front["val_mse"].idxmin(), "genome_id"]
    high_ret = front.loc[front["val_return"].idxmax(), "genome_id"]
    return {"Pareto choice": int(pick), "lowest val MSE end": int(low_mse), "highest val return end": int(high_ret)}


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    cache_path = WORK / "pareto_search_trades.npz"
    cache = dict(np.load(cache_path)) if cache_path.exists() else {}
    vname = {s: f"V{i + 1}" for i, s in enumerate(SETS)}

    def traded(y, s, r, gid, split, P):
        """Daily returns and days traded of one genome on one split (cached)."""
        key = f"{RUN}|{y}|{s}|{r}|{gid}|{split}"
        if key not in cache:
            st = ret_stats(s)
            ret, n = paper_returns(prediction_dir(y, f"pareto{RUN_TAG}_{s}_g{gid}", r, s, split,
                                                  P * st["iqr"] + st["median"]), price_dir(y, s, split))
            cache[key], cache[f"{key}|trades"] = ret, np.array([n, len(ret)])
        return cache[key], int(cache[f"{key}|trades"][0])

    genome_rows, missing = [], []
    for y in YEARS:
        for s in SETS:
            fronts = {}
            for r in REPEATS:
                d = EXAMM / "test_output" / f"{RUN}_{y}" / s / r
                if (d / "pareto_front.csv").exists():
                    fronts[r] = pd.read_csv(d / "pareto_front.csv").assign(run=r)
                else:
                    missing.append(f"{y} {s}/{r}: no pareto_front.csv")
            if not fronts:
                continue
            pooled = pd.concat(fronts.values(), ignore_index=True)
            for r, front in fronts.items():
                d = EXAMM / "test_output" / f"{RUN}_{y}" / s / r
                picks = choose(front, pooled)
                for _, g in front.iterrows():
                    gid = int(g["genome_id"])
                    Pv = read_predictions(d / f"evaluation_g{gid}_val", "val")
                    Pt = read_predictions(d / f"evaluation_g{gid}_test", "test")
                    if Pv is None or Pt is None or not (np.isfinite(Pv.to_numpy()).all()
                                                        and np.isfinite(Pt.to_numpy()).all()):
                        missing.append(f"{y} {s}/{r} genome {gid}: no complete forecasts")
                        continue
                    vr, _ = traded(y, s, r, gid, "val", Pv)
                    tr, tdays = traded(y, s, r, gid, "test", Pt)
                    val_mse, val_ic = forecast_scores(Pv, y, s, "val")
                    test_mse, test_ic = forecast_scores(Pt, y, s, "test")
                    mt = metrics(tr)
                    genome_rows.append({
                        "fold": y, "test_year": y, "set": s, "panel": vname[s], "repeat": r, "genome_id": gid,
                        "front_size": len(front),
                        "chosen_as": ", ".join(c for c in CHOICES if picks[c] == gid),
                        "val_mse_examm": g["val_mse"], "val_return_examm_%": g["val_return"] * 100,
                        "val_return_%": total_return(vr), "val_mse": val_mse, "val_rank_ic": val_ic,
                        "total_return_%": total_return(tr), "sharpe": mt["sharpe"], "max_dd_%": mt["max_dd_%"],
                        "test_mse": test_mse, "test_rank_ic": test_ic, "test_ic_pearson": pearson_ic(Pt, y, s, "test"),
                        "test_days_traded": tdays,
                        "enabled_nodes": g.get("enabled_nodes"), "enabled_edges": g.get("enabled_edges"),
                        "enabled_recurrent_edges": g.get("enabled_recurrent_edges")})
                np.savez(cache_path, **cache)
    np.savez(cache_path, **cache)

    if missing:
        print(f"!! {len(missing)} item(s) left out: " + "; ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))
        print("   forecasts: sh scripts/rnn_examples/evaluate_rnn_core7_pareto.sh (in EXAMM-Extended)")
    if not genome_rows:
        sys.exit("nothing to report")
    G = pd.DataFrame(genome_rows)

    # 1. validation check: Python's validation return of every genome vs the val_return EXAMM wrote
    G["val_check_diff_pp"] = (G["val_return_%"] - G["val_return_examm_%"]).abs()
    off = G[G["val_check_diff_pp"] > CHECK_TOLERANCE * 100]
    print(f"\nValidation check: {len(G)} genome(s); Python vs EXAMM validation return, largest difference "
          f"{G['val_check_diff_pp'].max():.2e} percentage points; {len(off)} above {CHECK_TOLERANCE * 100:g} pp")
    G.to_csv(RESULTS / f"{OUT}_genomes.csv", index=False)
    if len(off) > 0.05 * len(G):
        print(off[["test_year", "set", "repeat", "genome_id", "val_return_examm_%", "val_return_%"]].head(20)
              .to_string(index=False))
        sys.exit(f"STOPPED: more than 5% of the genomes disagree with EXAMM's validation return -- the search and the "
                 f"backtest do not score genomes the same way; check the trading files and forecasts first "
                 f"(details in {RESULTS / f'{OUT}_genomes.csv'})")
    if len(off):
        print(f"   ({len(off)} genome(s) differ by more than the tolerance, under 5%: forecast rounding can flip a "
              f"close ranking; listed in {OUT}_genomes.csv, column val_check_diff_pp)")

    # 2-3. per run and test year: the chosen genome and the two front ends
    rows = []
    for _, g in G[G["chosen_as"] != ""].iterrows():
        for c in CHOICES:
            if c in g["chosen_as"].split(", "):
                rows.append({"design": "retrain every year", "choice": c, **{k: g[k] for k in (
                    "fold", "test_year", "set", "panel", "repeat", "genome_id", "total_return_%", "sharpe",
                    "max_dd_%", "test_mse", "test_rank_ic", "test_ic_pearson", "val_return_%", "val_mse",
                    "front_size", "test_days_traded")}})
    R = pd.DataFrame(rows).sort_values(["choice", "test_year", "set", "repeat"])
    R.to_csv(RESULTS / f"{OUT}_repeats.csv", index=False)

    pick = R[R["choice"] == "Pareto choice"]
    n_runs = pick.groupby(["test_year", "set"]).size()
    print(f"\nMEAN OVER THE RUNS per panel and test year ({RUN}, retrain every year, L/S {K}/{K}, costs on); "
          f"runs per cell: {n_runs.min()}-{n_runs.max()}")
    for c in CHOICES:
        t = R[R["choice"] == c].pivot_table(index="test_year", columns="panel", values="total_return_%", aggfunc="mean")
        t["mean"] = t.mean(axis=1)
        print(f"\n{c}: test net return % (mean over runs)\n" + t.round(2).to_string())
    print(f"\nValidation vs test, Pareto choice (mean over all runs): validation return "
          f"{pick['val_return_%'].mean():.2f} %, test return {pick['total_return_%'].mean():.2f} %")
    print(f"\nSaved {RESULTS / f'{OUT}_genomes.csv'} and {RESULTS / f'{OUT}_repeats.csv'}")
    print("Next: python3 core7_pareto_search_compare.py" + ("" if RUN_TAG == "" else f" --run {RUN}"))


if __name__ == "__main__":
    main()
