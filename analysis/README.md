# analysis/ — the core7 backtest and reporting scripts (Financial_toolbox)

These scripts turn EXAMM's core7 forecasts into trading results, tables and figures. On the Mac they are run from a
copy of this folder, `~/Financial_toolbox`; the EXAMM side (data, runs, forecasts) lives in `~/EXAMM-Extended`.

**Start here:**
- [`MAIN_WORKFLOW.md`](MAIN_WORKFLOW.md) — the main experiment, step by step (forecasts → backtest → figures → workbook).
- [`RET_protocol.md`](RET_protocol.md) — the frozen evaluation protocol of the earlier RET results.
- [`RUN_GENOME_FRONTIER.md`](RUN_GENOME_FRONTIER.md) — the genome Pareto-frontier side experiment.

## Paths

Scripts find their inputs relative to these locations (environment variables override the defaults):

| Variable | Default | Used for |
|---|---|---|
| `EXAMM_DIR` | `~/EXAMM-Extended` | `datasets/` (fold files, `panels_core7`) and `test_output/` (EXAMM runs and their `evaluation_*` forecasts) |
| `CORE7_PANELS` | `$EXAMM_DIR/datasets/panels_core7` | `<set>_core7/panel_dates.csv` (prices, raw returns, costs) and `panel_meta.json` (RET median / IQR) |
| `FIN_RESULTS_DIR` | `results/` next to the scripts | output folder (`core7_repeats.py`, `main_core7.py`; the other scripts always write to `results/`) |

Intermediate files (rebuilt price / prediction files, trade caches such as `results/genome_frontier/*.npz`) are
written under `results/` and can be deleted at any time; delete the `*.npz` trade caches after any change to the
trading code or the forecasts, or the scripts reuse the old trades.

## Files

**Trading engine**
- `portfolio.py` — `Portfolio`: the trading rules (below), run over a set of `Stock`s.
- `stock.py` — `Stock`: one position, its prices and transaction costs. With costs on (`use_TC`), a buy pays
  price + `TRAN_COST` and a sale receives price − `TRAN_COST`, on both the long and the short side (opening a short is a
  sale, covering it is a buy).
- `paper_long_short.py` — the paper's Algorithm 1 (daily long-short, rebalanced only when the k-th highest forecast is
  positive and the k-th lowest negative), built on the `Stock` methods.
- `ensemble.py` — combining several models' forecasts (`rank_mean`) and their agreement (`mean_pairwise_spearman`).
- `Logger.py` — logging used by the engine.

**Main experiment** (see `MAIN_WORKFLOW.md`)
- `core7_main_results.py` — the backtest and report: best seed, mean over the 10 seeds, `--pareto` validation choice.
- `plot_core7_alpha.py`, `plot_core7_pareto_corr.py`, `plot_core7_main_table5.py` — figures from its CSVs.
- `core7_results_sections.py` — the Results-section tables and tests.
- `make_core7_main_workbook.py` — the Excel workbook.
- `core7_genome_frontier.py` — shared helpers (prices, forecasts, metrics, bootstrap, Pareto) used by the above.

**Side experiments**
- `core7_genome_frontier*.py`, `make_genome_frontier_workbook.py` — genome Pareto frontier (`RUN_GENOME_FRONTIER.md`).
- `core7_mo_frontier.py` — frontier of the multi-objective EXAMM runs.
- `core7_frontier.py`, `main_core7.py`, `main_core7_protocol.py`, `make_original_rules_workbook.py`, `core7_table.py`,
  `core7_repeats.py` — the earlier RET protocol (`RET_protocol.md`), rule comparisons and per-repeat results.
- `core7_paper_tables.py` — workbook of the selection-year experiment.

**Checks**
- `check_cpp_trading_metric.py` — confirms EXAMM's in-search trading return (C++, `rnn/trading_metric.cxx`) equals
  `paper_long_short.py` on the same forecasts (run from the EXAMM-Extended directory).

## Trading rules in `portfolio.py`

- `daily_long_short_return` — long the top L, short the bottom S forecasts, rebalanced every day.
- `conditional_long_short_return` — the same, but rebalanced only on days when the forecasts separate the
  cross-section by sign (the paper's Algorithm 1; `paper_long_short.py` is the version the main results use).
- `overlapping_return` — Jegadeesh-Titman overlapping books: each day a new long-short book is formed and held
  `hold_days`, so `hold_days` books are open at once, each with `1/hold_days` of the capital; books are netted, so only
  the change in a name's total position pays the spread (`Stock.rebalance_to`).
- `banded_return` — enter a name when its forecast rank reaches `entry_rank`, hold until it falls past `exit_rank`;
  the band between the two avoids churn from small rank changes.
- `daily_equal_weight_return`, `buy_and_hold_return` — benchmarks; `short_term_reversal_return` — the reversal
  benchmark; `simple_return`, `portfolio_simple_return`, `daily_long_return`, `portfolio_shorting_return`,
  `daily_hybrid_long_short_return`, `long_short_return` — earlier rules, kept for reproducing old results.
