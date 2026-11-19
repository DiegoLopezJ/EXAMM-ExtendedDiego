// The validation trading return of one genome, as EXAMM computes it during the search (--fitness_metric return):
// the genome's forecasts on the validation files, turned into raw-return forecasts and traded with the paper's
// long-short rule, costs on (rnn/trading_metric.hxx).
//
//   evaluate_trading_return --genome_file <bin> --testing_filenames <dir>/*_val.csv --time_offset 1
//                           --trading_file <dir>/trading_validation.csv [--trading_k 10] [--write_forecasts <csv>]
//
// Prints the net return (and the days traded). --write_forecasts saves the raw forecasts at full precision (one
// column per stock) so analysis/check_cpp_trading_metric.py can trade the same numbers with paper_long_short.py.

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
using std::string;
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/trading_metric.hxx"
#include "time_series/time_series.hxx"

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);
    if (!argument_exists(arguments, "--output_directory")) {
        arguments.push_back("--output_directory");
        arguments.push_back(".");
    }
    if (!argument_exists(arguments, "--std_message_level")) {
        arguments.push_back("--std_message_level");
        arguments.push_back("ERROR");
    }
    if (!argument_exists(arguments, "--file_message_level")) {
        arguments.push_back("--file_message_level");
        arguments.push_back("NONE");
    }
    Log::initialize(arguments);
    Log::set_id("main");

    string genome_filename, trading_file;
    get_argument(arguments, "--genome_file", true, genome_filename);
    get_argument(arguments, "--trading_file", true, trading_file);
    vector<string> testing_filenames;
    get_argument_vector(arguments, "--testing_filenames", true, testing_filenames);
    int32_t time_offset = 1, k = 10;
    get_argument(arguments, "--time_offset", true, time_offset);
    get_argument(arguments, "--trading_k", false, k);

    RNN_Genome* genome = new RNN_Genome(genome_filename);
    TimeSeriesSets* time_series_sets = TimeSeriesSets::generate_test(
        testing_filenames, genome->get_input_parameter_names(), genome->get_output_parameter_names()
    );
    string normalize_type = genome->get_normalize_type();
    if (normalize_type.compare("min_max") == 0) {
        time_series_sets->normalize_min_max(genome->get_normalize_mins(), genome->get_normalize_maxs());
    } else if (normalize_type.compare("avg_std_dev") == 0) {
        time_series_sets->normalize_avg_std_dev(
            genome->get_normalize_avgs(), genome->get_normalize_std_devs(), genome->get_normalize_mins(),
            genome->get_normalize_maxs()
        );
    }
    vector<vector<vector<double> > > inputs, outputs;
    time_series_sets->export_test_series(time_offset, inputs, outputs);

    TradingData data;
    string error;
    if (!load_trading_data(trading_file, testing_filenames, data, error)) {
        Log::fatal("ERROR: %s\n", error.c_str());
        return 1;
    }
    set_validation_trading_data(data, k);

    vector<double> parameters = genome->get_best_parameters();
    double through_genome = genome->get_trading_return(parameters, inputs, outputs);

    // the same, step by step, to report the days traded and save the forecasts
    vector<vector<double> > predictions = genome->get_predictions(parameters, inputs, outputs);
    vector<vector<double> > raw(predictions.size());
    for (int32_t i = 0; i < (int32_t) predictions.size(); i++) {
        for (double p : predictions[i]) {
            raw[i].push_back(genome->denormalize_output(0, p) * data.ret_iqr + data.ret_median);
        }
    }
    TradingResult result = paper_long_short(raw, data.prices, data.costs, k, true, data.stocks);

    string forecasts_file;
    if (get_argument(arguments, "--write_forecasts", false, forecasts_file)) {
        std::ofstream out(forecasts_file);
        out << std::setprecision(17);
        for (int32_t i = 0; i < (int32_t) data.stocks.size(); i++) {
            out << (i ? "," : "") << data.stocks[i];
        }
        out << "\n";
        for (int32_t t = 0; t < (int32_t) raw[0].size(); t++) {
            for (int32_t i = 0; i < (int32_t) raw.size(); i++) {
                out << (i ? "," : "") << raw[i][t];
            }
            out << "\n";
        }
    }

    printf("trading_return %.17g\n", through_genome);
    printf("trading_return_direct %.17g\n", result.total_return);
    printf("days_traded %d of %d\n", result.days_traded, result.days > 0 ? result.days - 1 : 0);
    printf("validation_mse %.17g\n", genome->get_mse(parameters, inputs, outputs));
    delete genome;
    return through_genome == result.total_return ? 0 : 2;
}
