# EXAMM-MO: multi-objective EXAMM (validation MSE vs information ratio, or vs portfolio return)

A copy of the EXAMM-Extended source with one added option. `EXAMM-Extended` itself is not changed: its build,
datasets, results and scripts are as they were. This folder reads the datasets from `EXAMM-Extended/datasets`
and writes its own build, benchmark files and results here.

## What the options do

| | Original EXAMM (`--objective mse`, the default) | `--objective mse_ir` | `--objective mse_ret` |
|---|---|---|---|
| Genome scores | validation MSE | validation MSE **and** validation information ratio (IR) | validation MSE **and** validation portfolio return |
| Island keeps / selects parents by | lowest MSE | Pareto rank on (lower MSE, higher IR), ties by crowding distance (NSGA-II) | Pareto rank on (lower MSE, higher return), same method |
| Island ranking for extinction | best MSE | best MSE (unchanged) | best MSE (unchanged) |
| Benchmark file | -- | `--ir_benchmark_file` (reversal strategy) | none |
| Saved at the end | `global_best_genome_<id>.bin` (lowest MSE) | the same **plus** `pareto_genome_<id>.bin` + `pareto_front.csv` (`val_mse, val_ir`) | the same, `pareto_front.csv` has `val_mse, val_return` |
| Job script / results | -- | `scripts/core7_examm_yearly_ret_mo.sh` -> `test_output/core7_examm_ret_mo_<Y>/` | `scripts/core7_examm_yearly_ret_mo_return.sh` -> `test_output/core7_examm_ret_mo_return_<Y>/` |

**Portfolio return** (`mse_ret`): the same daily long-short 5/5 book as below; its annualized mean daily
return (mean x 252), before costs, in EXAMM's normalized RET units (the raw return times one positive constant
shared by all genomes, so genomes rank exactly as on raw returns; real % returns with costs come from the Mac
backtest).

**IR**, computed by each worker after training, at the weights EXAMM keeps (best validation MSE epoch):
each validation day, long the 5 stocks with the highest forecast and short the 5 lowest; day return = mean
realized RET of the longs minus the shorts (before costs). The book is regressed on the reversal benchmark,
r = a + b·bench + e, and IR = a / std(e) × √252. The benchmark is the 10-day z-score reversal strategy traded
the same way on the same days (`benchmarks/…/reversal_ls55_val.csv`).

With `--objective mse` (the default) the code behaves exactly as the original and writes byte-identical
genome files. Multi-objective genome files are the original format plus a 12-byte block at the end
("MOBJ" + IR), so the ordinary `evaluate_rnn` (EXAMM-Extended, Mac) reads them unchanged.

## Changed files (`git diff 5a1d9c5` shows every line)

- `rnn/rnn_genome.hxx/.cxx`: `best_validation_obj2` (the second objective), `selection_score`,
  `get_validation_book()`, `get_validation_ir()`, `get_validation_return()`, `get_validation_objective()`, `dominates()`,
  static multi-objective settings; IR computed at the end of `backpropagate_stochastic`; IR copied in `copy()`
  and carried in the binary format (needed for MPI: workers send genomes back to the master as bytes).
- `examm/island.hxx/.cxx`: `insert_genome_multi_objective()`, `rank_pareto()`; best/worst island fitness = MSE.
- `examm/island_speciation_strategy.hxx/.cxx`, `examm/speciation_strategy.hxx`: global Pareto archive,
  `save_pareto_archive()`.
- `examm/examm.cxx`: saves the archive at the end of a run.
- `mpi/examm_mpi.cxx`: options `--objective mse|mse_ir|mse_ret`, `--ir_benchmark_file` (mse_ir), `--ir_long_short_n` (5),
  `--pareto_archive_size` (50); every rank loads the benchmark.

New files: `data_prep/write_core7_reversal_benchmark.py`, `scripts/core7_examm_yearly_ret_mo.sh`,
`scripts/core7_examm_yearly_ret_mo_return.sh`,
`scripts/select_core7_mo_candidates.py`, `scripts/evaluate_rnn_core7_ret_mo_cross.sh`.

## Workflow

1. Benchmark files (done for 2022-2024): `python3 data_prep/write_core7_reversal_benchmark.py --years 2022 2023 2024`
2. Build (done): `build/mpi/examm_mpi`, same compiler / MPI as EXAMM-Extended (openmpi 4.1.6, gcc 11.2.0).
3. **Small test run (needs approval: sbatch)**:
   `sbatch --array=1 --ntasks=16 -t 0:45:00 --export=ALL,YEAR=2022,MO_TEST=1 scripts/core7_examm_yearly_ret_mo.sh`
   -> `test_output/core7_examm_ret_mo_test_2022/set1/15/` (check `pareto_front.csv`, `fitness_log.csv`).
4. **Full runs (needs approval: sbatch)**, 3 years x 4 sets x 10 repeats:
   `sbatch --array=1-4 --export=ALL,YEAR=<Y> scripts/core7_examm_yearly_ret_mo.sh` for Y = 2022, 2023, 2024.
   MSE vs return: the same with `scripts/core7_examm_yearly_ret_mo_return.sh` (no benchmark needed).
5. Candidates per fold: `python3 scripts/select_core7_mo_candidates.py --years 2022 2023 2024`
   (MSE vs return: add `--name core7_examm_ret_mo_return`)
   (per source set, the non-dominated genomes of its 10 runs + each run's lowest-MSE genome).
6. Mac: copy `test_output/core7_examm_ret_mo_<Y>/` into `~/EXAMM-Extended/test_output/`, then
   `sh evaluate_rnn_core7_ret_mo_cross.sh` (every candidate on all 4 sets, val and test;
   MSE vs return: `RUN=core7_examm_ret_mo_return sh evaluate_rnn_core7_ret_mo_cross.sh`).
7. Mac: genome frontier on the candidates (validation choice, test reported once) -- script to be written
   once the runs exist.

## Checks done (offline, no search run)

- An original genome re-written with the option off is byte-identical; its stored validation MSE is
  reproduced exactly from the validation files.
- The C++ IR of an existing genome matches an independent Python calculation (difference < 1e-6).
- IR survives the MPI byte-array and file round trips; the multi-objective file = original bytes + IR block.
- Island of 10 fed 40 genomes: every globally non-dominated genome survives, order and structure map consistent.
- Archive = exactly the non-dominated set; with a cap it keeps both ends of the front.
- Candidate selection and evaluation scripts run on a mock folder (stand-in evaluate_rnn); reruns skip finished work.

- MSE vs return: the C++ return objective matches Python (difference 2e-15); after adding it, the IR of the
  test genome is unchanged (2.2757205131); the return survives the MPI round trip; `pareto_front.csv` names
  its column `val_ir` or `val_return`; candidate selection reads both.

Not yet checked: a real `mpirun` search (step 3).

## Fix (2026-11-05): placeholders during island initialization

The first MSE vs IR runs (jobs 21940080-82) rejected the untrained placeholder genomes EXAMM inserts while an
island initializes. Islands then stayed empty, every new genome was an unmutated copy of the seed, and the
placeholders still counted toward `--max_genomes`: the runs trained only 67-96% of the genomes of the original
MSE-only runs, and 2023 set1 (where the seed's training returns NaN) trained 2% and stopped after 17 s.
Fixed in `examm/island.cxx`: placeholders are inserted as in the original EXAMM (every trained genome dominates
them, so they are removed first) and are left out of the crowding ranges. Results from before the fix are not
comparable with the MSE-only runs and should be rerun.

## Fix (2026-11-07): master crash in `RNN_Genome::add_edge`

The rerun (jobs 21977264-66) crashed 42 times, all in the master process: `EXAMM::mutate` -> `RNN_Genome::add_edge`
indexed an empty node list when a genome picked for mutation had no reachable node (or all reachable nodes at one
depth). This is the original EXAMM's known add_edge segfault (its guard was commented out); the multi-objective
islands keep more varied genomes, so it happened far more often (2023: 9 repeats incomplete, 2 tasks timed out
retrying). Fixed with the guard: add_edge returns false and `mutate` tries another mutation. It is reached only
where the old code crashed, so completed runs are unaffected and stay comparable.

## Option `--book_hold_days H` (2026-11-07): search objective matched to the overlapping rule

`--book_hold_days 10` makes the in-search long-short book hold each day's picks 10 days (day t's return = average
over the books opened on days t-9..t), matching the overlap 5/5 hold 10 trading rule. Default 1 = the daily book
(unchanged: the IR and return of the test genome are reproduced exactly). Job script:
`scripts/core7_examm_yearly_ret_mo_ret10.sh` (`--objective mse_ret --book_hold_days 10`) ->
`test_output/core7_examm_ret_mo_ret10_<Y>/`. Mac frontier: `core7_mo_frontier.py --run ret10 --rule overlap`.
Check: C++ hold-10 book matches an independent Python calculation (max daily difference 6e-17).
