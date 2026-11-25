#!/bin/bash -l
#SBATCH -J examm_core7_exact_eval
#SBATCH -A cis251123
#SBATCH -p wholenode
#SBATCH --ntasks=128
#SBATCH -t 4:00:00
#SBATCH -o /anvil/scratch/x-dlopez6/experiments/examm_results/examm_%x_%A.output
#SBATCH -e /anvil/scratch/x-dlopez6/experiments/examm_results/examm_%x_%A.error
#SBATCH --mail-user=rsch_lopedieg@kean.edu
#SBATCH --mail-type=ALL
#
# Re-evaluates the core7 genomes with EXACTLY the normalization their search used (evaluate_rnn
# --normalization_files: the statistics recomputed from the fold's *_train.csv then *_val.csv, the order the job
# scripts pass them). Genome files written before the full-precision fix store the statistics rounded to 6 digits,
# so plain evaluate_rnn feeds the networks inputs about 1e-7 off; unstable networks then give different forecasts.
#
# Writes the forecasts into the same folders the Mac pipeline reads (existing forecasts there are replaced):
#   core7-ret   EXAMM-Extended/test_output/core7_examm_ret_<F>/<set>/<repeat>/  (final global best genome)
#                 evaluation_val/       fold F's validation year      (core7_main_results.py --pareto)
#                 evaluation_test/      test year F                   (retrain every year, mean over seeds)
#                 evaluation_test_y<Y>/ later test years Y > F        (never retrain)
#   Pareto      EXAMM-refactor/test_output/core7_examm_ret_pareto_<F>/<set>/<repeat>/  (every front genome + global best)
#                 evaluation_g<id>_val/, evaluation_g<id>_test/
# Each finished folder gets a marker file NORMALIZATION_EXACT; a rerun skips those, so resubmitting only redoes what
# is missing. Runs ${SLURM_NTASKS} evaluations at a time (one evaluate_rnn per core).
#   sbatch scripts/rnn_examples/evaluate_core7_exact.sh                              (both experiments)
#   sbatch --export=ALL,ONLY=ret    scripts/rnn_examples/evaluate_core7_exact.sh    (core7-ret only)
#   sbatch --export=ALL,ONLY=pareto scripts/rnn_examples/evaluate_core7_exact.sh    (Pareto only)
# Do not change the locale (LC_ALL / LC_COLLATE) for this job: it decides the file order, as in the original runs.

module load gcc/11.2.0 openmpi/4.1.6

EXAMM="${EXAMM_DIR:-/anvil/scratch/x-dlopez6/experiments/EXAMM-Extended}"      # data and core7-ret runs
REFACTOR="${REFACTOR_DIR:-/anvil/scratch/x-dlopez6/experiments/EXAMM-refactor}"  # build and Pareto runs
EVALUATE="${EVALUATE:-$REFACTOR/build/rnn_examples/evaluate_rnn}"
YEARS="${YEARS:-2022 2023 2024}"
SETS="${SETS:-set1 set2 set3 set4}"
REPEATS="${REPEATS:-15 16 17 18 19 20 21 22 23 24}"
ONLY="${ONLY:-all}"
PARALLEL="${SLURM_NTASKS:-8}"

if [ ! -x "$EVALUATE" ]; then
    echo "no $EVALUATE -- build EXAMM-refactor first" >&2; exit 1
fi

# one line per evaluation: genome | output folder | fold data folder (normalization) | evaluated data folder | split
tasks=$(mktemp)
add_task() {   # genome out fold_dir data_dir split
    [ -f "$2/NORMALIZATION_EXACT" ] && return
    echo "$1|$2|$3|$4|$5" >> "$tasks"
}
final_best() { ls "$1"/global_best_genome_*.bin 2>/dev/null | sed 's/.*global_best_genome_\([0-9]*\)\.bin$/\1/' | sort -n | tail -1; }

missing=0
for F in $YEARS; do
    for s in $SETS; do
        fold="$EXAMM/datasets/panels_core7_examm_$F/$s"
        if ! ls "$fold"/*_train.csv >/dev/null 2>&1; then echo "!! no training files in $fold" >&2; missing=$((missing + 1)); continue; fi
        for r in $REPEATS; do
            if [ "$ONLY" != "pareto" ]; then
                run="$EXAMM/test_output/core7_examm_ret_$F/$s/$r"
                id=$(final_best "$run")
                if [ -z "$id" ]; then echo "!! core7-ret $F $s/$r: no global_best_genome" >&2; missing=$((missing + 1))
                else
                    g="$run/global_best_genome_$id.bin"
                    add_task "$g" "$run/evaluation_val" "$fold" "$fold" val
                    add_task "$g" "$run/evaluation_test" "$fold" "$fold" test
                    for Y in $YEARS; do
                        [ "$Y" -gt "$F" ] && add_task "$g" "$run/evaluation_test_y$Y" "$fold" "$EXAMM/datasets/panels_core7_examm_$Y/$s" test
                    done
                fi
            fi
            if [ "$ONLY" != "ret" ]; then
                run="$REFACTOR/test_output/core7_examm_ret_pareto_$F/$s/$r"
                if [ ! -f "$run/pareto_front.csv" ]; then echo "!! Pareto $F $s/$r: no pareto_front.csv" >&2; missing=$((missing + 1))
                else
                    ids=$( (tail -n +2 "$run/pareto_front.csv" | cut -d, -f1; final_best "$run") | sort -un)
                    for id in $ids; do
                        if [ -f "$run/pareto_genome_$id.bin" ]; then g="$run/pareto_genome_$id.bin"; else g="$run/global_best_genome_$id.bin"; fi
                        add_task "$g" "$run/evaluation_g${id}_val" "$fold" "$fold" val
                        add_task "$g" "$run/evaluation_g${id}_test" "$fold" "$fold" test
                    done
                fi
            fi
        done
    done
done

evaluate_one() {   # one task line
    IFS='|' read -r genome out fold data split <<< "$1"
    rm -rf "$out"; mkdir -p "$out"
    n_expected=$(ls "$data"/*_${split}.csv | wc -l)
    "$EVALUATE" --genome_file "$genome" \
        --testing_filenames "$data"/*_${split}.csv \
        --normalization_files "$fold"/*_train.csv "$fold"/*_val.csv \
        --time_offset 1 --output_directory "$out" \
        --std_message_level ERROR --file_message_level NONE > "$out/evaluate.err" 2>&1
    n=$(ls "$out"/*_${split}_predictions.csv 2>/dev/null | wc -l)
    if [ "$n" -eq "$n_expected" ]; then
        rm -f "$out/evaluate.err"; touch "$out/NORMALIZATION_EXACT"; echo ok
    else
        echo "FAILED $genome -> $out ($n of $n_expected files): $(head -c 300 "$out/evaluate.err")" >&2; echo failed
    fi
}
export -f evaluate_one
export EVALUATE

total=$(wc -l < "$tasks")
echo "$total evaluations to do ($PARALLEL at a time); $missing item(s) missing"
results=$(xargs -d '\n' -P "$PARALLEL" -I{} bash -c 'evaluate_one "$1"' _ {} < "$tasks")
rm -f "$tasks"
ok=$(echo "$results" | grep -c '^ok$'); failed=$(echo "$results" | grep -c '^failed$')
echo "evaluated $ok, failed $failed, missing $missing (already done before this run were skipped)"
[ "$failed" -eq 0 ] && [ "$missing" -eq 0 ]
