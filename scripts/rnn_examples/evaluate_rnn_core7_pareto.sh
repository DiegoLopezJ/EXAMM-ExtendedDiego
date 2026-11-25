#!/bin/sh
# PARETO SEARCH, step 1 (forecasts): every genome the multi-objective runs saved, on its own set's validation and
# test files, so the Python backtest can check and trade them (analysis/PARETO_SEARCH_PROTOCOL.md).
#
# Runs: test_output/<RUN>_<Y>/<set>/<repeat>/ from scripts/base_run/core7_examm_yearly_ret_pareto.sh
#   RUN=core7_examm_ret_pareto (default) or RUN=core7_examm_ret_pareto_ckret
# Genomes per run: every pareto_genome_<id>.bin listed in pareto_front.csv, plus the run's final global best
# (highest-numbered global_best_genome_<id>.bin, the lowest-validation-MSE genome) if it is not already among them.
# Each genome is evaluated on datasets/panels_core7_examm_<Y>/<set>/*_val.csv and *_test.csv (retrain every year:
# the genome of fold Y on validation Y-1 and test Y; the test files carry the usual 40-day warm-up, which the
# backtest drops). Output: <repeat>/evaluation_g<id>_val/ and <repeat>/evaluation_g<id>_test/ with one
# <STOCK>_<split>_predictions.csv per stock. A folder that already holds all of them is skipped, so a rerun only
# evaluates what is missing.
#
# Usage (from anywhere; paths are relative to this EXAMM-Extended checkout):
#   sh scripts/rnn_examples/evaluate_rnn_core7_pareto.sh
#   RUN=core7_examm_ret_pareto_ckret YEARS="2024" SETS="set2" sh scripts/rnn_examples/evaluate_rnn_core7_pareto.sh
# Ends with "evaluated N, already done M, failed F, missing X"; exit status 0 only if F and X are 0.

E=$(cd "$(dirname "$0")/../.." && pwd)
RUN=${RUN:-core7_examm_ret_pareto}
YEARS=${YEARS:-"2022 2023 2024"}
SETS=${SETS:-"set1 set2 set3 set4"}
REPEATS=${REPEATS:-"15 16 17 18 19 20 21 22 23 24"}
EVALUATE=${EVALUATE:-$E/build/rnn_examples/evaluate_rnn}

if [ ! -x "$EVALUATE" ]; then
    echo "!! no evaluate_rnn at $EVALUATE -- build EXAMM first (or set EVALUATE=<path>)"
    exit 1
fi

done_n=0; skip_n=0; fail_n=0; missing_n=0
for Y in $YEARS; do
    for s in $SETS; do
        data=$E/datasets/panels_core7_examm_$Y/$s
        n_stocks=$(ls "$data"/*_val.csv 2>/dev/null | wc -l | tr -d ' ')
        if [ "$n_stocks" = "0" ] || ! ls "$data"/*_test.csv >/dev/null 2>&1; then
            echo "!! no validation / test files in $data -- copy the fold files first"
            missing_n=$((missing_n + 1)); continue
        fi
        for r in $REPEATS; do
            run=$E/test_output/${RUN}_$Y/$s/$r
            if [ ! -f "$run/pareto_front.csv" ]; then
                echo "!! $RUN $Y $s/$r: no pareto_front.csv -- copy the finished run from Anvil"
                missing_n=$((missing_n + 1)); continue
            fi
            ids=$(tail -n +2 "$run/pareto_front.csv" | cut -d, -f1)
            best=$(ls "$run"/global_best_genome_*.bin 2>/dev/null \
                   | sed 's/.*global_best_genome_\([0-9]*\)\.bin$/\1/' | sort -n | tail -1)
            seen=""
            for id in $ids $best; do
                case " $seen " in *" $id "*) continue ;; esac   # the global best is usually also on the front
                seen="$seen $id"
                if [ -f "$run/pareto_genome_$id.bin" ]; then genome=$run/pareto_genome_$id.bin
                elif [ -f "$run/global_best_genome_$id.bin" ]; then genome=$run/global_best_genome_$id.bin
                else
                    echo "!! $RUN $Y $s/$r: genome $id listed but no .bin file"; missing_n=$((missing_n + 1)); continue
                fi
                for split in val test; do
                    out=$run/evaluation_g${id}_$split
                    n=$(ls "$out"/*_${split}_predictions.csv 2>/dev/null | wc -l | tr -d ' ')
                    if [ "$n" = "$n_stocks" ]; then skip_n=$((skip_n + 1)); continue; fi
                    mkdir -p "$out"
                    echo "== $RUN $Y $s/$r genome $id on $split"
                    "$EVALUATE" \
                    --testing_filenames "$data"/*_${split}.csv \
                    --time_offset 1 \
                    --genome_file "$genome" \
                    --output_directory "$out" \
                    --std_message_level ERROR \
                    --file_message_level ERROR
                    n=$(ls "$out"/*_${split}_predictions.csv 2>/dev/null | wc -l | tr -d ' ')
                    if [ "$n" = "$n_stocks" ]; then done_n=$((done_n + 1)); else
                        echo "!! $RUN $Y $s/$r genome $id on $split: only $n of $n_stocks prediction files written"
                        fail_n=$((fail_n + 1)); fi
                done
            done
        done
    done
done
echo "evaluated $done_n, already done $skip_n, failed $fail_n, missing $missing_n"
[ $fail_n -eq 0 ] && [ $missing_n -eq 0 ]
