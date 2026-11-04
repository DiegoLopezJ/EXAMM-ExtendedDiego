# EXAMM-MO: multi-objective EXAMM (validation MSE vs information ratio)

A copy of the EXAMM-Extended source with one added option. `EXAMM-Extended` itself is not changed: its build,
datasets, results and scripts are as they were. This folder reads the datasets from `EXAMM-Extended/datasets`
and writes its own build, benchmark files and results here.

## What `--objective mse_ir` does

| | Original EXAMM (`--objective mse`, the default) | `--objective mse_ir` |
|---|---|---|
| Genome scores | validation MSE | validation MSE **and** validation information ratio (IR) |
| Island keeps / selects parents by | lowest MSE | Pareto rank on (lower MSE, higher IR), ties by crowding distance (NSGA-II) |
| Island ranking for extinction | best MSE | best MSE (unchanged) |
| Saved at the end | `global_best_genome_<id>.bin` (lowest MSE) | the same **plus** `pareto_genome_<id>.bin` + `pareto_front.csv` (global Pareto archive, up to `--pareto_archive_size`) |

**IR**, computed by each worker after training, at the weights EXAMM keeps (best validation MSE epoch):
each validation day, long the 5 stocks with the highest forecast and short the 5 lowest; day return = mean
realized RET of the longs minus the shorts (before costs). The book is regressed on the reversal benchmark,
r = a + b·bench + e, and IR = a / std(e) × √252. The benchmark is the 10-day z-score reversal strategy traded
the same way on the same days (`benchmarks/…/reversal_ls55_val.csv`).

With `--objective mse` (the default) the code behaves exactly as the original and writes byte-identical
genome files. Multi-objective genome files are the original format plus a 12-byte block at the end
("MOBJ" + IR), so the ordinary `evaluate_rnn` (EXAMM-Extended, Mac) reads them unchanged.

## Changed files (`git diff 5a1d9c5` shows every line)

- `rnn/rnn_genome.hxx/.cxx`: `best_validation_ir`, `selection_score`, `get_validation_ir()`, `dominates()`,
  static multi-objective settings; IR computed at the end of `backpropagate_stochastic`; IR copied in `copy()`
  and carried in the binary format (needed for MPI: workers send genomes back to the master as bytes).
- `examm/island.hxx/.cxx`: `insert_genome_multi_objective()`, `rank_pareto()`; best/worst island fitness = MSE.
- `examm/island_speciation_strategy.hxx/.cxx`, `examm/speciation_strategy.hxx`: global Pareto archive,
  `save_pareto_archive()`.
- `examm/examm.cxx`: saves the archive at the end of a run.
- `mpi/examm_mpi.cxx`: options `--objective mse|mse_ir`, `--ir_benchmark_file`, `--ir_long_short_n` (5),
  `--pareto_archive_size` (50); every rank loads the benchmark.

New files: `data_prep/write_core7_reversal_benchmark.py`, `scripts/core7_examm_yearly_ret_mo.sh`,
`scripts/select_core7_mo_candidates.py`, `scripts/evaluate_rnn_core7_ret_mo_cross.sh`.

## Workflow

1. Benchmark files (done for 2022-2024): `python3 data_prep/write_core7_reversal_benchmark.py --years 2022 2023 2024`
2. Build (done): `build/mpi/examm_mpi`, same compiler / MPI as EXAMM-Extended (openmpi 4.1.6, gcc 11.2.0).
3. **Small test run (needs approval: sbatch)**:
   `sbatch --array=1 --ntasks=16 -t 0:45:00 --export=ALL,YEAR=2022,MO_TEST=1 scripts/core7_examm_yearly_ret_mo.sh`
   -> `test_output/core7_examm_ret_mo_test_2022/set1/15/` (check `pareto_front.csv`, `fitness_log.csv`).
4. **Full runs (needs approval: sbatch)**, 3 years x 4 sets x 10 repeats:
   `sbatch --array=1-4 --export=ALL,YEAR=<Y> scripts/core7_examm_yearly_ret_mo.sh` for Y = 2022, 2023, 2024.
5. Candidates per fold: `python3 scripts/select_core7_mo_candidates.py --years 2022 2023 2024`
   (per source set, the non-dominated genomes of its 10 runs + each run's lowest-MSE genome).
6. Mac: copy `test_output/core7_examm_ret_mo_<Y>/` into `~/EXAMM-Extended/test_output/`, then
   `sh evaluate_rnn_core7_ret_mo_cross.sh` (every candidate on all 4 sets, val and test).
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

Not yet checked: a real `mpirun` search (step 3).
