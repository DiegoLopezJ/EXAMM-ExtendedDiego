# Evaluation protocol: Pareto EXAMM search (validation MSE and validation return)

**Fixed before any test result of these runs was computed.** The rule below is applied unchanged; any other choice
(another selection rule, pooling across runs, a different trading rule) is reported as an exploratory addition and
never replaces the result produced here.

## 1. The runs

- **Search:** EXAMM with `--objectives mse,return` (`scripts/base_run/core7_examm_yearly_ret_pareto.sh`). Islands
  keep and select genomes by Pareto rank on (lower validation MSE, higher validation return), ties by crowding
  distance (NSGA-II). Validation return = the paper's Algorithm 1 (long 10 / short 10, rebalance only when the
  10th-highest forecast > 0 and the 10th-lowest < 0), costs on both legs, on the validation year
  (`rnn/trading_metric.cxx`, equal to `paper_long_short.py` to 1.7e-14).
- **Everything else as core7-ret:** core7 panels set1–set4, folds test year Y ∈ {2022, 2023, 2024} (train ≤ Y−2,
  validation Y−1, test Y with a 40-day warm-up), the same 7 inputs, RET target, 10 islands × 10, 10,000 genomes,
  20 epochs, repeats 15–24. Training loss MSE; each network keeps its lowest-validation-MSE epoch.
- **Saved per run:** the Pareto archive (`pareto_genome_<id>.bin`, `pareto_front.csv` with each genome's validation
  MSE and return) and the global best by validation MSE.

## 2. Genome choice (validation only)

Per run (one set, one fold, one repeat), among the genomes of **that run's own** `pareto_front.csv`:

- **Pareto choice (main result):** the genome with the lowest sum of its rank by validation MSE (ascending) and its
  rank by validation return (descending), both ranked over the fronts of **all 10 runs** of that panel and fold
  (`method="min"`); ties → the lower validation MSE. This is the rule of the after-the-fact `--pareto` choice in
  `core7_main_results.py`, which ranks over the 10 repeats. (Ranked within one front only, the sum is the same for
  every genome, since a front ordered by MSE is ordered by return in reverse, and the tie-break would always pick
  the lowest-MSE end; the pooled ranks avoid that.)
- **Front ends (reported alongside, not chosen):** the lowest validation MSE genome and the highest validation return
  genome of the run's front.

Only the validation numbers in `pareto_front.csv` (written by EXAMM during the search) are used to choose. Before
choosing, every genome's validation return is recomputed in Python from its `evaluate_rnn` forecasts and must match
`pareto_front.csv` (`core7_pareto_search_results.py` stops otherwise).

## 3. Test (reported once)

- **Design:** retrain every year: the fold-Y runs' chosen genomes are traded on test year Y only.
- **Trading:** `paper_long_short.py` with K = 10, sign check, transaction costs on (the corrected `stock.py`: shorts
  sell at price − cost and cover at price + cost), initial capital 1000, as the main core7-ret result.
- **Per run and test year:** total net return over the year (%), Sharpe, max drawdown, test MSE (raw daily-return
  units), Rank IC and Pearson IC — the columns of `core7_main_repeats.csv`.
- **Reported number:** the mean over the 10 runs (seeds) per panel set and test year, the paper's reporting; the same
  for the two front ends.

## 4. Comparison with core7-ret

- **Baseline:** `core7_main_repeats.csv`, design "retrain every year" (the mean over the 10 core7-ret seeds), from
  `core7_main_results.py` run with the corrected `stock.py`.
- **Same budget:** 10 runs × 10,000 genomes per set and fold in both experiments; only what evolution keeps differs.
- **Tests:** per set and year, two-sided Mann-Whitney U on the 10 vs 10 runs' test net returns; over the 12
  set-years, the paired Wilcoxon signed-rank test on the per-set-year means.
- **Caveat stated with the results:** the search used the validation year to keep networks that trade well, so
  validation returns of these runs are optimistic; only test numbers are evidence. The validation-versus-test gap is
  reported.
