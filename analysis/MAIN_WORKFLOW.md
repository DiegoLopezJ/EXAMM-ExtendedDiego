# Main workflow (core7, Pooled EXAMM RET, paper's trading rule and reporting)

Design: per test year Y (2022, 2023, 2024) train <= Y-2, validate Y-1, test Y; 10 EXAMM runs (seeds) per panel set
V1-V4; traded with the paper's Algorithm 1 (long 10 / short 10, rebalance only when the 10th-highest forecast > 0 and
the 10th-lowest < 0, forecasts converted to raw returns, costs on). --k 5 = the 5/5 book.
Main comparison with the paper = the MEAN OVER THE 10 SEEDS (as the paper reports); the single best seed by validation
MSE is shown next to it as a labelled second row.

1. Data and EXAMM runs (Anvil, done): scripts/data_prep/split_core7_yearly.py, scripts/base_run/core7_examm_yearly_ret.sh
2. Forecasts (Mac, in ~/EXAMM-Extended; folders already done are skipped):
        sh scripts/rnn_examples/evaluate_rnn_core7_main.sh                  best seed, own + later years
        sh scripts/rnn_examples/evaluate_rnn_core7_ret_cross.sh             every seed, own validation + test year
        ALL_REPEATS=1 sh scripts/rnn_examples/evaluate_rnn_core7_main.sh    every seed, later years (never retrain)
3. Backtest (Mac, in ~/Financial_toolbox):
        python3 core7_main_results.py --pareto            everything: best seed, MEAN OVER THE 10 SEEDS, Pareto choice
   options: --k 5 (5/5 book, outputs end in _k5), --no-mean-repeats, --trade-every-day (no sign check),
            --all (leakage cells, for comparison with the paper only)
4. Extra figures (Mac):
        python3 plot_core7_alpha.py           Figure 1 / Table 6: net alpha vs MKT + STR1 + STR5 (Driscoll-Kraay)
        python3 plot_core7_pareto_corr.py     does a validation score predict the test result?
   (both take --k 5)
        python3 core7_results_sections.py     Results section tables: per set and year (retrain / never retrain),
                                              aggregates, per-panel Sharpe and drawdown, selection objectives,
                                              Mann-Whitney U and paired Wilcoxon tests -> results/core7_results_sections.xlsx
   (also takes --k 5; its T2 test needs both the 10/10 and the 5/5 results, i.e. core7_main_results.py run with and
   without --k 5)
5. Workbook (Mac):    python3 make_core7_main_workbook.py && open results/core7_main.xlsx      (--k 5 -> core7_main_k5.xlsx)
   Summary / Per set / Table 5 / Table 8: paper (blue), yours MEAN OF 10 SEEDS (yellow, bold), best seed (grey, italic).
   Repeats T7/T8: the mean over seeds in full. Grid / Joined / Rule compared / Model size: the best seed.

Main result: "retrain every year" (the paper's design). "Never retrain" and the Pareto choice are robustness checks.
The paper's IC (+0.0106, Tables 4 and 8) is the PEARSON IC -- compare it with the IC (Pearson) columns.
Paper: arXiv 2610.07825. Still different from it: the inputs (paper: Return, Volume Change, Bid-Ask Spread, Illiquidity,
Turnover, S&P 500 Return; raw values), 10 backprop epochs (here 20), and the target scaling (here robust z-score);
matching them needs raw CRSP data for the 200 stocks, which is not available.
Code: paper_long_short.py (the paper's Algorithm 1), core7_genome_frontier.py (shared helper functions).
Side experiments (not the main result): core7_genome_frontier*.py (Pareto genome frontier), core7_mo_frontier.py
(multi-objective EXAMM), core7_frontier.py (trading-rule comparison), main_core7_protocol.py (trade every day).

## Pareto EXAMM search (validation MSE and return) — see PARETO_SEARCH_PROTOCOL.md
1. Runs (Anvil, EXAMM-refactor): scripts/base_run/core7_examm_yearly_ret_pareto.sh (--objectives mse,return), then copy
   test_output/core7_examm_ret_pareto_<Y>/ to ~/EXAMM-Extended/test_output/ (quote the remote path in rsync).
2. Forecasts (Mac, in ~/EXAMM-Extended): sh scripts/rnn_examples/evaluate_rnn_core7_pareto.sh   (every front genome, val + test)
3. Backtest (Mac, in ~/Financial_toolbox): python3 core7_pareto_search_results.py   (validation check, choice, test)
4. Comparison: python3 core7_pareto_search_compare.py -> results/core7_pareto_search_vs_ret.xlsx
   (needs results/core7_main_repeats.csv from core7_main_results.py run with the corrected stock.py)

## Exact normalization for evaluation (applies to core7-ret and the Pareto runs)
Genome files written before commit "Store normalization statistics at full precision" hold the data-normalization
statistics rounded to 6 digits, so plain evaluate_rnn fed the networks inputs about 1e-7 off from the search; unstable
networks then gave different, saturated forecasts (11 of 120 core7-ret genomes, 55 of 1,429 Pareto genomes).
Re-evaluate with the search's exact statistics (evaluate_rnn --normalization_files <fold train files> <fold val files>):
1. Anvil: sbatch scripts/rnn_examples/evaluate_core7_exact.sh   (both experiments; writes the usual evaluation_* folders)
2. Copy the evaluation_* folders to the Mac, delete results/genome_frontier/ (cached forecasts and trades), then rerun
   core7_main_results.py --pareto, core7_pareto_search_results.py and core7_pareto_search_compare.py.
Genomes written after the fix store the exact statistics, and plain evaluate_rnn reproduces the search.
