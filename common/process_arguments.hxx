#ifndef PROCESS_ARGUMENTS_HXX
#define PROCESS_ARGUMENTS_HXX

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "examm/examm.hxx"
#include "examm/island_speciation_strategy.hxx"
#include "examm/neat_speciation_strategy.hxx"
#include "rnn/rnn_genome.hxx"
#include "time_series/time_series.hxx"

EXAMM* generate_examm_from_arguments(
    const vector<string>& arguments, TimeSeriesSets* time_series_sets, WeightRules* weight_rules,
    RNN_Genome* seed_genome
);
SpeciationStrategy* generate_speciation_strategy_from_arguments(
    const vector<string>& arguments, RNN_Genome* seed_genome
);
IslandSpeciationStrategy* generate_island_speciation_strategy_from_arguments(
    const vector<string>& arguments, RNN_Genome* seed_genome
);
NeatSpeciationStrategy* generate_neat_speciation_strategy_from_arguments(
    const vector<string>& arguments, RNN_Genome* seed_genome
);

/**
 * Normalizes the time series a saved genome is evaluated on (evaluate_rnn, evaluate_trading_return).
 *
 * Default: with the statistics stored in the genome file (the original behavior). Genome files written before the
 * statistics were saved at full precision hold them rounded to 6 significant digits, so the evaluated inputs differ
 * from the ones the search used by about 1e-7; stable networks are unaffected, unstable ones can give completely
 * different forecasts.
 *
 * --normalization_files <files>: recompute the statistics exactly as the search did, from the same files in the
 * same order (the run's --training_filenames, then its --validation_filenames), and use them for the evaluated series
 * and for the genome's output denormalization. Exits if they are far from the genome's stored statistics (wrong
 * files). Returns true if the statistics were recomputed.
 */
bool normalize_for_evaluation(const vector<string>& arguments, RNN_Genome* genome, TimeSeriesSets* evaluated);

bool is_island_strategy(string strategy_name);
bool is_neat_strategy(string strategy_name);
void set_island_transfer_learning_parameters(
    const vector<string>& arguments, IslandSpeciationStrategy* island_strategy
);

void write_time_series_to_file(const vector<string>& arguments, TimeSeriesSets* time_series_sets);
void get_train_validation_data(
    const vector<string>& arguments, TimeSeriesSets* time_series_sets, vector<vector<vector<double> > >& traing_inputs,
    vector<vector<vector<double> > >& train_outputs, vector<vector<vector<double> > >& test_inputs,
    vector<vector<vector<double> > >& test_outputs
);
void slice_input_data(
    vector<vector<vector<double> > >& traing_inputs, vector<vector<vector<double> > >& train_outputs,
    int32_t sequence_length
);
vector<vector<double> > slice_time_series(
    int32_t start_index, int32_t sequence_length, int32_t num_parameter, const vector<vector<double> >& time_series
);

#endif