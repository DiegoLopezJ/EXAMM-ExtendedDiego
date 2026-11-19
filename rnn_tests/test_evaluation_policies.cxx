// Tests for the separated evaluation responsibilities (training loss / checkpoint selection / fitness):
//   1. defaults: a new genome trains with MSE, keeps the lowest-validation-MSE weights and uses validation MSE as
//      fitness; the default loss reproduces the established MSE gradient scaling exactly
//   2. gradient checks: analytic (backprop) vs numerical gradients for MSE and a custom loss, on recurrent networks
//      with one and several outputs
//   3. checkpoint selection follows its own metric
//   4. fitness follows its own metric, independently of checkpoint selection
//   5. changing checkpoint selection / fitness does not change training; a custom loss does
//   6. the policies survive serialization (the path genomes take between the MPI coordinator and its workers), and
//      genomes written before the policies existed still load with the MSE defaults
// Exits with status 1 if any check fails.
//
//   ./rnn_tests/test_evaluation_policies [--std_message_level INFO]

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>
using std::string;
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "rnn/evaluation_metric.hxx"
#include "rnn/generate_nn.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/training_loss.hxx"
#include "weights/weight_rules.hxx"
#include "weights/weight_update.hxx"

typedef vector<vector<double> > Series;
typedef vector<Series> SeriesSet;

static int32_t failures = 0;
static int32_t checks = 0;

static void check(bool condition, const string& what) {
    checks++;
    if (!condition) {
        failures++;
        Log::error("FAILED: %s\n", what.c_str());
    } else {
        Log::debug("passed: %s\n", what.c_str());
    }
}

static std::mt19937 generator(20261119);

static Series random_series(int32_t n_parameters, int32_t length) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    Series s(n_parameters, vector<double>(length));
    for (auto& row : s) {
        for (auto& v : row) {
            v = u(generator);
        }
    }
    return s;
}

static vector<double> random_parameters(int32_t n) {
    std::uniform_real_distribution<double> u(-0.5, 0.5);
    vector<double> p(n);
    for (auto& v : p) {
        v = u(generator);
    }
    return p;
}

// A custom, smooth loss with a plain gradient (both scales 1): sum over outputs of mean_j log(cosh(y - t)).
class LogCoshLoss : public TrainingLoss {
   public:
    string get_name() const override {
        return "test_log_cosh";
    }
    double compute(const Series& predicted, const Series& expected, Series& output_errors) const override {
        output_errors.resize(expected.size());
        double sum = 0.0;
        for (size_t i = 0; i < expected.size(); i++) {
            output_errors[i].resize(expected[i].size());
            double total = 0.0;
            double n = (double) expected[i].size();
            for (size_t j = 0; j < expected[i].size(); j++) {
                double d = predicted[i][j] - expected[i][j];
                total += std::log(std::cosh(d));
                output_errors[i][j] = std::tanh(d) / n;
            }
            sum += total / n;
        }
        return sum;
    }
};

// MSE with the textbook gradient 2 (y - t) / T and no extra scaling, to pin down EXAMM's established scaling:
// the default MSE gradient must equal loss x this gradient.
class PlainMSELoss : public TrainingLoss {
   public:
    string get_name() const override {
        return "test_plain_mse";
    }
    double compute(const Series& predicted, const Series& expected, Series& output_errors) const override {
        output_errors.resize(expected.size());
        double sum = 0.0;
        for (size_t i = 0; i < expected.size(); i++) {
            output_errors[i].resize(expected[i].size());
            double total = 0.0;
            double n = (double) expected[i].size();
            for (size_t j = 0; j < expected[i].size(); j++) {
                double d = predicted[i][j] - expected[i][j];
                total += d * d;
                output_errors[i][j] = 2.0 * d / n;
            }
            sum += total / n;
        }
        return sum;
    }
};

static LogCoshLoss log_cosh_loss;
static PlainMSELoss plain_mse_loss;

static bool close(double a, double b, double abs_tol, double rel_tol) {
    return std::fabs(a - b) <= abs_tol + rel_tol * std::max(std::fabs(a), std::fabs(b));
}

// ---------------------------------------------------------------------------------------------------------------
// 1 + 2: defaults, established scaling, gradient checks
// ---------------------------------------------------------------------------------------------------------------

static void test_gradients(const string& name, RNN_Genome* genome, int32_t length) {
    genome->set_stochastic(false);
    int32_t n_in = genome->get_number_inputs();
    int32_t n_out = genome->get_number_outputs();
    RNN* rnn = genome->get_rnn();

    for (int32_t attempt = 0; attempt < 3; attempt++) {
        Series inputs = random_series(n_in, length);
        Series outputs = random_series(n_out, length);
        vector<double> parameters = random_parameters(rnn->get_number_weights());

        vector<double> analytic, empirical, explicit_mse, plain;
        double loss_a, loss_e, loss_x, loss_p;

        // MSE (default): analytic vs numerical, and the default equals the explicit MSE loss bit for bit
        rnn->get_analytic_gradient(parameters, inputs, outputs, loss_a, analytic, false, true, 0.0);
        rnn->get_empirical_gradient(parameters, inputs, outputs, loss_e, empirical, false, true, 0.0);
        rnn->get_analytic_gradient(
            parameters, inputs, outputs, loss_x, explicit_mse, false, true, 0.0, get_training_loss("mse")
        );
        bool ok = loss_a == loss_e && loss_a == loss_x && analytic == explicit_mse;
        for (size_t j = 0; j < analytic.size(); j++) {
            ok = ok && close(analytic[j], empirical[j], 1e-9, 1e-6);
        }
        check(ok, name + ": MSE analytic gradient matches the numerical gradient and the explicit MSE loss");

        // established scaling: default MSE gradient == loss * (textbook MSE gradient)
        rnn->get_analytic_gradient(parameters, inputs, outputs, loss_p, plain, false, true, 0.0, &plain_mse_loss);
        ok = close(loss_p, loss_a, 0.0, 1e-12);
        for (size_t j = 0; j < analytic.size(); j++) {
            ok = ok && close(analytic[j], loss_a * plain[j], 1e-14, 1e-9);
        }
        check(ok, name + ": default MSE keeps the established gradient scaling (loss x dMSE/dw)");

        // custom loss: analytic vs numerical
        rnn->get_analytic_gradient(parameters, inputs, outputs, loss_a, analytic, false, true, 0.0, &log_cosh_loss);
        rnn->get_empirical_gradient(parameters, inputs, outputs, loss_e, empirical, false, true, 0.0, &log_cosh_loss);
        ok = loss_a == loss_e;
        for (size_t j = 0; j < analytic.size(); j++) {
            ok = ok && close(analytic[j], empirical[j], 1e-8, 1e-6);
        }
        check(ok, name + ": custom loss analytic gradient matches the numerical gradient");
    }
    delete rnn;
}

static void test_defaults(WeightRules* weight_rules) {
    RNN_Genome* genome = create_lstm(vector<string>{"a", "b"}, 1, 2, vector<string>{"y"}, 2, weight_rules);
    check(genome->get_training_loss() == default_training_loss(), "default training loss is the default loss");
    check(genome->get_training_loss()->get_name() == "mse", "default training loss is mse");
    check(genome->get_checkpoint_metric() == EvaluationMetric::MSE, "default checkpoint metric is mse");
    check(genome->get_fitness_metric() == EvaluationMetric::MSE, "default fitness metric is mse");
    RNN_Genome* copy = genome->copy();
    check(
        copy->get_training_loss() == genome->get_training_loss()
            && copy->get_checkpoint_metric() == genome->get_checkpoint_metric()
            && copy->get_fitness_metric() == genome->get_fitness_metric(),
        "copy() keeps the evaluation policies"
    );
    delete copy;
    delete genome;

    EvaluationMetric m = EvaluationMetric::MAE;
    check(parse_evaluation_metric("MSE", m) && m == EvaluationMetric::MSE, "parse 'MSE'");
    check(parse_evaluation_metric("mae", m) && m == EvaluationMetric::MAE, "parse 'mae'");
    check(!parse_evaluation_metric("rmse", m) && m == EvaluationMetric::MAE, "reject 'rmse'");
    check(get_training_loss("no_such_loss") == nullptr, "unknown training loss is not found");
}

// ---------------------------------------------------------------------------------------------------------------
// 3 + 4: checkpoint selection and fitness follow their own metrics
// ---------------------------------------------------------------------------------------------------------------

static void test_checkpoint_and_fitness(WeightRules* weight_rules) {
    RNN_Genome* genome = create_elman(vector<string>{"a", "b"}, 1, 3, vector<string>{"y1", "y2"}, 2, weight_rules);
    genome->set_stochastic(false);
    int32_t n = genome->get_number_weights();

    SeriesSet validation_inputs{random_series(2, 30), random_series(2, 30)};
    SeriesSet validation_outputs{random_series(2, 30), random_series(2, 30)};

    // two parameter vectors that the metrics rank differently: A has the lower MSE, B the lower MAE
    vector<vector<double> > candidates;
    vector<double> mses, maes;
    for (int32_t i = 0; i < 200; i++) {
        candidates.push_back(random_parameters(n));
        mses.push_back(genome->get_mse(candidates.back(), validation_inputs, validation_outputs));
        maes.push_back(genome->get_mae(candidates.back(), validation_inputs, validation_outputs));
    }
    int32_t a = -1, b = -1;
    for (int32_t i = 0; i < 200 && a < 0; i++) {
        for (int32_t j = 0; j < 200; j++) {
            if (mses[i] < mses[j] && maes[i] > maes[j]) {
                a = i;
                b = j;
                break;
            }
        }
    }
    check(a >= 0, "found two weight vectors ranked differently by MSE and MAE");
    if (a < 0) {
        delete genome;
        return;
    }

    const EvaluationMetric metrics[2] = {EvaluationMetric::MSE, EvaluationMetric::MAE};
    for (EvaluationMetric checkpoint : metrics) {
        for (EvaluationMetric fitness : metrics) {
            string tag =
                "checkpoint " + evaluation_metric_name(checkpoint) + ", fitness " + evaluation_metric_name(fitness);
            RNN_Genome* g = genome->copy();
            g->set_checkpoint_metric(checkpoint);
            g->set_fitness_metric(fitness);

            // epoch 1 sees A, epoch 2 sees B
            check(
                g->update_checkpoint(candidates[a], mses[a], validation_inputs, validation_outputs),
                tag + ": the first epoch is always kept"
            );
            bool kept_b = g->update_checkpoint(candidates[b], mses[b], validation_inputs, validation_outputs);

            int32_t expected = checkpoint == EvaluationMetric::MSE ? a : b;
            check(
                kept_b == (expected == b), tag + ": the second epoch is kept only if it improves the checkpoint metric"
            );
            check(
                g->get_best_parameters() == candidates[expected], tag + ": kept weights follow the checkpoint metric"
            );
            check(
                g->get_best_validation_mse() == mses[expected] && g->get_best_validation_mae() == maes[expected],
                tag + ": recorded validation MSE and MAE are those of the kept weights"
            );
            double expected_fitness = fitness == EvaluationMetric::MSE ? mses[expected] : maes[expected];
            check(g->get_fitness() == expected_fitness, tag + ": fitness follows the fitness metric");
            delete g;
        }
    }
    delete genome;
}

// ---------------------------------------------------------------------------------------------------------------
// 5 + 6: training is unaffected by checkpoint / fitness; serialization
// ---------------------------------------------------------------------------------------------------------------

static RNN_Genome* round_trip(RNN_Genome* genome) {
    char* bytes;
    int32_t length;
    genome->write_to_array(&bytes, length);
    RNN_Genome* copy = new RNN_Genome(bytes, length);
    free(bytes);
    return copy;
}

// training MSE and validation MSE per epoch, from the genome's log file
static vector<string> epoch_losses(const string& log_filename) {
    vector<string> rows;
    std::ifstream in(log_filename);
    string line;
    std::getline(in, line);  // header
    while (std::getline(in, line)) {
        // iteration, milliseconds, training mse, validation mse, ...
        size_t c1 = line.find(',');
        size_t c2 = line.find(',', c1 + 1);
        size_t c3 = line.find(',', c2 + 1);
        size_t c4 = line.find(',', c3 + 1);
        rows.push_back(line.substr(c2 + 1, c4 - c2 - 1));
    }
    return rows;
}

static void train(
    RNN_Genome* genome, const string& log_filename, const SeriesSet& ti, const SeriesSet& to, const SeriesSet& vi,
    const SeriesSet& vo
) {
    WeightUpdate weight_update;
    genome->set_log_filename(log_filename);
    genome->backpropagate_stochastic(ti, to, vi, vo, &weight_update);
}

static void test_training_and_serialization(WeightRules* weight_rules, const string& directory) {
    RNN_Genome* base = create_jordan(vector<string>{"a", "b"}, 1, 3, vector<string>{"y1", "y2"}, 3, weight_rules);
    base->initialize_randomly();
    vector<double> initial;
    base->get_weights(initial);
    base->set_initial_parameters(initial);
    base->set_bp_iterations(8);

    SeriesSet ti{random_series(2, 40), random_series(2, 40), random_series(2, 40)};
    SeriesSet to{random_series(2, 40), random_series(2, 40), random_series(2, 40)};
    SeriesSet vi{random_series(2, 40), random_series(2, 40)};
    SeriesSet vo{random_series(2, 40), random_series(2, 40)};

    // serialized copies keep the random generator, so all of them shuffle the training series identically
    RNN_Genome* mse_genome = round_trip(base);
    RNN_Genome* mae_genome = round_trip(base);
    RNN_Genome* custom_genome = round_trip(base);
    mae_genome->set_checkpoint_metric(EvaluationMetric::MAE);
    mae_genome->set_fitness_metric(EvaluationMetric::MAE);
    custom_genome->set_training_loss(&log_cosh_loss);

    train(mse_genome, directory + "/policy_test_mse.csv", ti, to, vi, vo);
    train(mae_genome, directory + "/policy_test_mae.csv", ti, to, vi, vo);
    train(custom_genome, directory + "/policy_test_custom.csv", ti, to, vi, vo);

    vector<string> mse_epochs = epoch_losses(directory + "/policy_test_mse.csv");
    vector<string> mae_epochs = epoch_losses(directory + "/policy_test_mae.csv");
    vector<string> custom_epochs = epoch_losses(directory + "/policy_test_custom.csv");
    check(mse_epochs.size() == 8, "training ran all epochs");
    check(
        mse_epochs == mae_epochs,
        "changing checkpoint selection and fitness to MAE leaves training (per-epoch train / validation MSE) unchanged"
    );
    check(mse_epochs != custom_epochs, "a custom training loss changes training");

    // the kept weights carry the recorded values
    check(
        mae_genome->get_best_validation_mae() == mae_genome->get_mae(mae_genome->get_best_parameters(), vi, vo)
            && mae_genome->get_best_validation_mse() == mae_genome->get_mse(mae_genome->get_best_parameters(), vi, vo),
        "MAE checkpoint: recorded validation MSE / MAE belong to the kept weights"
    );
    check(mae_genome->get_fitness() == mae_genome->get_best_validation_mae(), "MAE fitness after training");
    check(mse_genome->get_fitness() == mse_genome->get_best_validation_mse(), "MSE fitness after training");
    check(
        mae_genome->get_best_validation_mae() <= mse_genome->get_best_validation_mae(),
        "the MAE checkpoint keeps weights with validation MAE no worse than the MSE checkpoint's"
    );
    check(
        mse_genome->get_best_validation_mse() <= mae_genome->get_best_validation_mse(),
        "the MSE checkpoint keeps weights with validation MSE no worse than the MAE checkpoint's"
    );

    // serialization (what examm_mpi sends between coordinator and workers)
    register_training_loss(&log_cosh_loss);
    custom_genome->set_checkpoint_metric(EvaluationMetric::MAE);
    custom_genome->set_fitness_metric(EvaluationMetric::MSE);
    RNN_Genome* received = round_trip(custom_genome);
    check(received->get_training_loss() == &log_cosh_loss, "serialization keeps a registered custom training loss");
    check(received->get_checkpoint_metric() == EvaluationMetric::MAE, "serialization keeps the checkpoint metric");
    check(received->get_fitness_metric() == EvaluationMetric::MSE, "serialization keeps the fitness metric");
    check(received->get_fitness() == custom_genome->get_fitness(), "serialization keeps the fitness value");
    delete received;

    received = round_trip(mae_genome);
    check(
        received->get_fitness_metric() == EvaluationMetric::MAE && received->get_fitness() == mae_genome->get_fitness()
            && received->get_best_parameters() == mae_genome->get_best_parameters(),
        "a worker's MAE-fitness genome reaches the coordinator with the same fitness and weights"
    );
    delete received;

    // a genome written before the policies existed: the same bytes without the trailing policy block
    char* bytes;
    int32_t length;
    mae_genome->write_to_array(&bytes, length);
    int32_t policy_bytes = 4 + 4 + (int32_t) string("mse").size() + 4 + 4;
    RNN_Genome* old_format = new RNN_Genome(bytes, length - policy_bytes);
    free(bytes);
    check(
        old_format->get_training_loss() == default_training_loss()
            && old_format->get_checkpoint_metric() == EvaluationMetric::MSE
            && old_format->get_fitness_metric() == EvaluationMetric::MSE,
        "a genome without policies (older file) loads with the MSE defaults"
    );
    check(
        old_format->get_best_parameters() == mae_genome->get_best_parameters()
            && old_format->get_best_validation_mse() == mae_genome->get_best_validation_mse()
            && old_format->get_fitness() == mae_genome->get_best_validation_mse(),
        "an older genome keeps its weights and is ranked by validation MSE as before"
    );
    delete old_format;

    delete base;
    delete mse_genome;
    delete mae_genome;
    delete custom_genome;
}

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);
    string directory = "./test_evaluation_policies_output";
    if (!argument_exists(arguments, "--output_directory")) {
        arguments.push_back("--output_directory");
        arguments.push_back(directory);
    } else {
        get_argument(arguments, "--output_directory", true, directory);
    }
    if (!argument_exists(arguments, "--std_message_level")) {
        arguments.push_back("--std_message_level");
        arguments.push_back("ERROR");
    }
    if (!argument_exists(arguments, "--file_message_level")) {
        arguments.push_back("--file_message_level");
        arguments.push_back("NONE");
    }
    mkdir(directory.c_str(), 0755);

    Log::initialize(arguments);
    Log::set_id("main");

    WeightRules* weight_rules = new WeightRules();
    weight_rules->initialize_from_args(arguments);

    test_defaults(weight_rules);

    vector<string> one_in{"a"}, two_in{"a", "b"}, one_out{"y"}, two_out{"y1", "y2"};
    for (int32_t depth = 1; depth <= 3; depth++) {
        string d = " (max recurrent depth " + std::to_string(depth) + ")";
        RNN_Genome* g;
        g = create_jordan(one_in, 1, 2, one_out, depth, weight_rules);
        test_gradients("Jordan 1x2, 1 output" + d, g, 12);
        delete g;
        g = create_jordan(two_in, 2, 2, two_out, depth, weight_rules);
        test_gradients("Jordan 2x2, 2 outputs" + d, g, 12);
        delete g;
        g = create_elman(two_in, 1, 3, two_out, depth, weight_rules);
        test_gradients("Elman 1x3, 2 outputs" + d, g, 12);
        delete g;
        g = create_lstm(two_in, 1, 2, two_out, depth, weight_rules);
        test_gradients("LSTM 1x2, 2 outputs" + d, g, 12);
        delete g;
        g = create_gru(one_in, 1, 2, two_out, depth, weight_rules);
        test_gradients("GRU 1x2, 2 outputs" + d, g, 12);
        delete g;
    }

    test_checkpoint_and_fitness(weight_rules);
    test_training_and_serialization(weight_rules, directory);

    delete weight_rules;
    printf("%d of %d checks passed\n", checks - failures, checks);
    if (failures > 0) {
        printf("SOME FAILED\n");
        return 1;
    }
    printf("ALL PASSED\n");
    return 0;
}
