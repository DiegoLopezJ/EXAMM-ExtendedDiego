#ifndef EXAMM_EVALUATION_METRIC_HXX
#define EXAMM_EVALUATION_METRIC_HXX

#include <cstdint>
#include <string>
using std::string;

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

#endif
