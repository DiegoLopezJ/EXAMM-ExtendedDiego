#include "rnn/evaluation_metric.hxx"

#include <algorithm>
#include <cctype>

string evaluation_metric_name(EvaluationMetric metric) {
    switch (metric) {
        case EvaluationMetric::MSE:
            return "mse";
        case EvaluationMetric::MAE:
            return "mae";
        case EvaluationMetric::RETURN:
            return "return";
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
    if (lower == "return") {
        metric = EvaluationMetric::RETURN;
        return true;
    }
    return false;
}

bool parse_evaluation_metric_list(const vector<string>& words, vector<EvaluationMetric>& metrics) {
    vector<EvaluationMetric> parsed;
    for (const string& word : words) {
        size_t start = 0;
        while (start <= word.size()) {
            size_t comma = word.find(',', start);
            string name = word.substr(start, comma == string::npos ? string::npos : comma - start);
            start = comma == string::npos ? word.size() + 1 : comma + 1;
            if (name.empty()) {
                continue;
            }
            EvaluationMetric metric;
            if (!parse_evaluation_metric(name, metric)) {
                return false;
            }
            for (EvaluationMetric m : parsed) {
                if (m == metric) {
                    return false;
                }
            }
            parsed.push_back(metric);
        }
    }
    metrics = parsed;
    return true;
}

namespace {
vector<EvaluationMetric> search_objectives;
}

void set_search_objectives(const vector<EvaluationMetric>& objectives) {
    search_objectives = objectives;
}

const vector<EvaluationMetric>& get_search_objectives() {
    return search_objectives;
}

bool is_multi_objective_search() {
    return search_objectives.size() >= 2;
}

bool is_valid_evaluation_metric(int32_t value) {
    return value == (int32_t) EvaluationMetric::MSE || value == (int32_t) EvaluationMetric::MAE
           || value == (int32_t) EvaluationMetric::RETURN;
}
