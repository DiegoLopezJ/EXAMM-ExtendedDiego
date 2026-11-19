#ifndef EXAMM_EVALUATION_METRIC_HXX
#define EXAMM_EVALUATION_METRIC_HXX

#include <cstdint>
#include <string>
using std::string;

#include <vector>
using std::vector;

/**
 * A validation metric, used for two independent decisions (each set separately on RNN_Genome):
 *   checkpoint selection: which weights from a genome's training history are kept (the epoch with the lowest value)
 *   evolutionary fitness: the scalar a trained genome is ranked by in the population (lower is better)
 * Neither changes the training loss (training_loss.hxx).
 *   MSE, MAE : validation error, lower is better
 *   RETURN   : validation net return of the paper's long-short rule with costs (trading_metric.hxx), higher is
 *              better; as fitness it is negated (EXAMM ranks lower fitness as better). Needs --trading_file.
 */
enum class EvaluationMetric : int32_t { MSE = 0, MAE = 1, RETURN = 2 };

string evaluation_metric_name(EvaluationMetric metric);

// "mse", "mae" or "return" (case-insensitive); returns false and leaves metric unchanged for anything else.
bool parse_evaluation_metric(const string& name, EvaluationMetric& metric);

// true for a value that is a valid EvaluationMetric (used when reading serialized genomes).
bool is_valid_evaluation_metric(int32_t value);

// A list such as "mse,return" (commas and/or separate words); false if a name is unknown or repeated.
bool parse_evaluation_metric_list(const vector<string>& words, vector<EvaluationMetric>& metrics);

/**
 * The search objectives (--objectives mse,return). With fewer than two, the search is single-objective: islands
 * rank genomes by get_fitness() as in the original EXAMM. With two or more, islands keep and select genomes by
 * Pareto rank on these validation metrics (NSGA-II: non-dominated fronts, ties by crowding distance) and a global
 * Pareto archive is kept and saved; get_fitness() is then the FIRST objective, used for the global best genome,
 * island extinction ranking and the logs. Set once at start-up, where the islands live (the coordinator).
 */
void set_search_objectives(const vector<EvaluationMetric>& objectives);
const vector<EvaluationMetric>& get_search_objectives();
bool is_multi_objective_search();

#endif
