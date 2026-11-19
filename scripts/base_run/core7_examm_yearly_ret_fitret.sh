#!/bin/bash -l
#SBATCH -J examm_core7_ret_fitret
#SBATCH -A cis251123
#SBATCH -p wholenode
#SBATCH --ntasks=128
#SBATCH -t 6:00:00
#SBATCH -o /anvil/scratch/x-dlopez6/experiments/examm_results/examm_%x_%A_%a.output
#SBATCH -e /anvil/scratch/x-dlopez6/experiments/examm_results/examm_%x_%A_%a.error
#SBATCH --mail-user=rsch_lopedieg@kean.edu
#SBATCH --mail-type=ALL
#
# RETURN-FITNESS version of core7_examm_yearly_ret.sh: same data, folds (train <= Y-2, val = Y-1, test = Y +
# 40-day warm-up), inputs, RET target, settings and repeats 15..24, but EXAMM ranks genomes by their VALIDATION
# NET RETURN instead of validation MSE:
#   --fitness_metric return     fitness = the paper's long-short 10/10 rule (sign check, costs on both legs) traded
#                               on the validation year with the genome's forecasts (rnn/trading_metric.hxx)
#   --trading_file              <fold>/<set>/trading_validation.csv (scripts/data_prep/write_core7_trading_files.py)
# Training is still MSE. Checkpoint selection (which epoch's weights each network keeps) is the lowest validation
# MSE, unless CHECKPOINT=return is exported (then the highest validation return).
# Uses the EXAMM-refactor build; reads the data from EXAMM-Extended; writes its results to
#   EXAMM-refactor/test_output/core7_examm_ret_fitret_<YEAR>/<set>/<repeat>/        (CHECKPOINT=mse, the default)
#   EXAMM-refactor/test_output/core7_examm_ret_fitret_ckret_<YEAR>/<set>/<repeat>/  (CHECKPOINT=return)
# so nothing mixes with the MSE runs in EXAMM-Extended/test_output/core7_examm_ret_<YEAR>/.
#   sbatch --array=1-4 --export=ALL,YEAR=2022 scripts/base_run/core7_examm_yearly_ret_fitret.sh
#   sbatch --array=1-4 --export=ALL,YEAR=2022,CHECKPOINT=return scripts/base_run/core7_examm_yearly_ret_fitret.sh
# Small test first (one set, one repeat, 500 genomes; output ..._test_<YEAR>):
#   sbatch --array=1 --ntasks=16 -t 0:45:00 --export=ALL,YEAR=2022,FITRET_TEST=1 scripts/base_run/core7_examm_yearly_ret_fitret.sh
# A folder whose run already finished is skipped, so resubmitting only redoes missing or incomplete repeats.

module load gcc/11.2.0 openmpi/4.1.6

EXAMM="/anvil/scratch/x-dlopez6/experiments/EXAMM-Extended"        # data only
REFACTOR="/anvil/scratch/x-dlopez6/experiments/EXAMM-refactor"     # build and results
YEAR="${YEAR:?set YEAR, e.g. sbatch --export=ALL,YEAR=2022 ...}"
SET="set${SLURM_ARRAY_TASK_ID:-1}"
FOLDERS="15 16 17 18 19 20 21 22 23 24"
MAX_GENOME=10000
CHECKPOINT="${CHECKPOINT:-mse}"
case "$CHECKPOINT" in
  mse)    RUN_NAME="core7_examm_ret_fitret" ;;
  return) RUN_NAME="core7_examm_ret_fitret_ckret" ;;
  *)      echo "CHECKPOINT must be mse or return, not '$CHECKPOINT'" >&2; exit 1 ;;
esac
if [ -n "$FITRET_TEST" ]; then
    FOLDERS="15"
    MAX_GENOME=500
    RUN_NAME="${RUN_NAME}_test"
fi
RUN_NAME="${RUN_NAME}_$YEAR"

DATAPATH="$EXAMM/datasets/panels_core7_examm_$YEAR/$SET"
TRADING_FILE="$DATAPATH/trading_validation.csv"
if ! ls $DATAPATH/*_train.csv >/dev/null 2>&1 || ! ls $DATAPATH/*_val.csv >/dev/null 2>&1; then
    echo "no train/val files in $DATAPATH -- run: python3 data_prep/split_core7_yearly.py --years $YEAR" >&2
    exit 1
fi
if [ ! -f "$TRADING_FILE" ]; then
    echo "no $TRADING_FILE -- run: python3 scripts/data_prep/write_core7_trading_files.py --years $YEAR" >&2
    exit 1
fi
if [ ! -x "$REFACTOR/build/mpi/examm_mpi" ]; then
    echo "no $REFACTOR/build/mpi/examm_mpi -- build EXAMM-refactor first" >&2
    exit 1
fi

INPUT_PARAMETERS="RET RET_CS_IN BA_SPREAD ILLIQUIDITY REV21_1 TURN_RATIO VOL21"
OUTPUT_PARAMETERS="RET"

failed=""
for FOLDER in $FOLDERS; do
    exp_name="$REFACTOR/test_output/$RUN_NAME/$SET/$FOLDER"

    # skip a repeat that already finished (fitness log reached max genomes)
    if [ -f "$exp_name/fitness_log.csv" ] && \
       [ "$(tail -1 "$exp_name/fitness_log.csv" | cut -d, -f1)" -ge $MAX_GENOME ] 2>/dev/null; then
        echo "$SET folder $FOLDER already finished, skipping"
        continue
    fi

    # retry up to 3 times if examm crashes; each attempt starts from an empty folder
    for attempt in 1 2 3; do
        rm -rf "$exp_name"
        mkdir -p "$exp_name"
        echo "Running EXAMM (fitness: validation return, checkpoint: $CHECKPOINT) on $SET (folder $FOLDER, attempt $attempt), results: $exp_name"
        echo "###-------------------###"

        # timeout: safety net in case a crashed run hangs; the return is one extra validation pass per genome
        # (every epoch with CHECKPOINT=return), so allow more than the MSE runs' 45 minutes
        time timeout 90m mpirun -np ${SLURM_NTASKS:-4} $REFACTOR/build/mpi/examm_mpi \
        --training_filenames $DATAPATH/*_train.csv \
        --validation_filenames $DATAPATH/*_val.csv \
        --trading_file $TRADING_FILE \
        --trading_k 10 \
        --fitness_metric return \
        --checkpoint_metric $CHECKPOINT \
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
echo "$SET: all folders $FOLDERS finished ($RUN_NAME)"
