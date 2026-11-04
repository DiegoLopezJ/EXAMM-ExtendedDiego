#!/bin/bash -l
#SBATCH -J examm_core7_ret_mo
#SBATCH -A cis251123
#SBATCH -p wholenode
#SBATCH --ntasks=128
#SBATCH -t 6:00:00
#SBATCH -o /anvil/scratch/x-dlopez6/experiments/examm_results/examm_%x_%A_%a.output
#SBATCH -e /anvil/scratch/x-dlopez6/experiments/examm_results/examm_%x_%A_%a.error
#SBATCH --mail-user=rsch_lopedieg@kean.edu
#SBATCH --mail-type=ALL
#
# MULTI-OBJECTIVE version of EXAMM-Extended/scripts/base_run/core7_examm_yearly_ret.sh.
# Same data, folds (train <= Y-2, val = Y-1, test = Y + 40-day warm-up), inputs, RET target, settings and
# repeats 15..24, but run with the EXAMM-MO build and
#   --objective mse_ir --ir_benchmark_file <EXAMM-MO>/benchmarks/panels_core7_examm_<Y>/<set>/reversal_ls55_val.csv
# so islands keep / select genomes by Pareto rank on (lower validation MSE, higher validation information
# ratio of a daily long-short 5/5 book against the 10-day z-score reversal strategy), and each run also saves
# its Pareto archive (pareto_genome_<id>.bin + pareto_front.csv) next to global_best_genome_<id>.bin.
# Benchmark files first (once): cd EXAMM-MO && python3 data_prep/write_core7_reversal_benchmark.py
# Results: EXAMM-MO/test_output/core7_examm_ret_mo_<YEAR>/<set>/<repeat>/ (EXAMM-Extended is not touched).
#   sbatch --array=1-4 --export=ALL,YEAR=2022 scripts/core7_examm_yearly_ret_mo.sh
#   sbatch --array=1-4 --export=ALL,YEAR=2023 scripts/core7_examm_yearly_ret_mo.sh
#   sbatch --array=1-4 --export=ALL,YEAR=2024 scripts/core7_examm_yearly_ret_mo.sh
# Small test first (one set, one repeat, 500 genomes, ~minutes; output core7_examm_ret_mo_test_<YEAR>):
#   sbatch --array=1 --ntasks=16 -t 0:45:00 --export=ALL,YEAR=2022,MO_TEST=1 scripts/core7_examm_yearly_ret_mo.sh
# A folder whose run already finished is skipped, so resubmitting only redoes missing or incomplete repeats.

EXAMM="/anvil/scratch/x-dlopez6/experiments/EXAMM-Extended"   # data only
EXAMM_MO="/anvil/scratch/x-dlopez6/experiments/EXAMM-MO"       # build, benchmarks, results
YEAR="${YEAR:?set YEAR, e.g. sbatch --export=ALL,YEAR=2022 ...}"
SET="set${SLURM_ARRAY_TASK_ID:-1}"
FOLDERS="15 16 17 18 19 20 21 22 23 24"
MAX_GENOME=10000
RUN_NAME="core7_examm_ret_mo_$YEAR"
if [ -n "$MO_TEST" ]; then
    FOLDERS="15"
    MAX_GENOME=500
    RUN_NAME="core7_examm_ret_mo_test_$YEAR"
fi
BENCHMARK="$EXAMM_MO/benchmarks/panels_core7_examm_$YEAR/$SET/reversal_ls55_val.csv"
if [ ! -f "$BENCHMARK" ]; then
    echo "missing $BENCHMARK -- run: cd $EXAMM_MO && python3 data_prep/write_core7_reversal_benchmark.py --years $YEAR" >&2
    exit 1
fi
DATAPATH="$EXAMM/datasets/panels_core7_examm_$YEAR/$SET"
if ! ls $DATAPATH/*_train.csv >/dev/null 2>&1 || ! ls $DATAPATH/*_val.csv >/dev/null 2>&1; then
    echo "no train/val files in $DATAPATH -- run: python3 data_prep/split_core7_yearly.py --years $YEAR" >&2
    exit 1
fi

INPUT_PARAMETERS="RET RET_CS_IN BA_SPREAD ILLIQUIDITY REV21_1 TURN_RATIO VOL21"
OUTPUT_PARAMETERS="RET"            # RET target (RET_CS runs: core7_examm_yearly.sh)

failed=""
for FOLDER in $FOLDERS; do
    exp_name="$EXAMM_MO/test_output/$RUN_NAME/$SET/$FOLDER"

    # skip a repeat that already finished (fitness log reached max genomes)
    if [ -f "$exp_name/fitness_log.csv" ] && \
       [ "$(tail -1 "$exp_name/fitness_log.csv" | cut -d, -f1)" -ge $MAX_GENOME ] 2>/dev/null; then
        echo "$SET folder $FOLDER already finished, skipping"
        continue
    fi

    # retry up to 3 times if examm crashes (segfault in RNN_Genome::add_edge, or
    # "mu or sigma exceeded possible bounds"); each attempt starts from an empty folder
    for attempt in 1 2 3; do
        rm -rf "$exp_name"
        mkdir -p "$exp_name"
        echo "Running EXAMM on $SET (folder $FOLDER, attempt $attempt), results will be saved to: "$exp_name
        echo "###-------------------###"

        # timeout: safety net in case a crashed run hangs (normal runs take ~12-20 min)
        time timeout 60m mpirun -np ${SLURM_NTASKS:-4} $EXAMM_MO/build/mpi/examm_mpi \
        --training_filenames $DATAPATH/*_train.csv \
        --validation_filenames $DATAPATH/*_val.csv \
        --time_offset 1 \
        --input_parameter_names $INPUT_PARAMETERS \
        --output_parameter_names $OUTPUT_PARAMETERS \
        --normalize avg_std_dev \
        --number_islands 10 \
        --island_size 10 \
        --max_genomes $MAX_GENOME \
        --bp_iterations 20 \
        --num_mutations 2 \
        --learning_rate 0.001 \
        --extinction_event_generation_number 500 \
        --repeat_extinction \
        --island_ranking_method "EraseWorst" \
        --repopulation_method "bestGenome" \
        --islands_to_exterminate 1 \
        --save_genome_option the_best \
        --objective mse_ir \
        --ir_benchmark_file $BENCHMARK \
        --ir_long_short_n 5 \
        --pareto_archive_size 50 \
        --possible_node_types simple UGRNN MGU GRU delta LSTM \
        --output_directory $exp_name \
        --std_message_level ERROR \
        --file_message_level ERROR && break

        echo "$SET folder $FOLDER: attempt $attempt failed" >&2
        if [ $attempt -eq 3 ]; then
            failed="$failed $FOLDER"
        fi
    done
done

if [ -n "$failed" ]; then
    echo "$SET folders that failed all 3 attempts:$failed" >&2
    exit 1
fi
echo "$SET: all folders $FOLDERS finished"
