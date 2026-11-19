#include "rnn/evaluation_metric.hxx"

#include <algorithm>
#include <cctype>

string evaluation_metric_name(EvaluationMetric metric) {
    switch (metric) {
        case EvaluationMetric::MSE:
            return "mse";
        case EvaluationMetric::MAE:
            return "mae";
    }
    return "unknown";
}

bool parse_evaluation_metric(const string& name, EvaluationMetric& metric) {
    string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
    if (lower == "mse") {
        metric = EvaluationMetric::MSE;
        return true;
    }
    if (lower == "mae") {
        metric = EvaluationMetric::MAE;
        return true;
    }
    return false;
}

bool is_valid_evaluation_metric(int32_t value) {
    return value == (int32_t) EvaluationMetric::MSE || value == (int32_t) EvaluationMetric::MAE;
}
