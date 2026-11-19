#include "rnn/training_loss.hxx"

#include <map>
#include <mutex>
using std::map;

string MSE_Loss::get_name() const {
    return "mse";
}

// Same arithmetic, in the same order, as RNN::calculate_error_mse, so default training is bit-for-bit unchanged.
double MSE_Loss::compute(
    const vector<vector<double> >& predicted, const vector<vector<double> >& expected,
    vector<vector<double> >& output_errors
) const {
    output_errors.resize(expected.size());

    double mse_sum = 0.0;
    for (int32_t i = 0; i < (int32_t) expected.size(); i++) {
        output_errors[i].resize(expected[i].size());

        double mse = 0.0;
        for (int32_t j = 0; j < (int32_t) expected[i].size(); j++) {
            double error = predicted[i][j] - expected[i][j];
            output_errors[i][j] = error;
            mse += error * error;
        }
        mse_sum += mse / expected[i].size();
    }
    return mse_sum;
}

double MSE_Loss::backward_scale(double loss, int32_t series_length) const {
    return loss * (1.0 / series_length) * 2.0;
}

double MSE_Loss::gradient_check_scale(double loss) const {
    return loss;
}

namespace {
std::mutex registry_mutex;

map<string, const TrainingLoss*>& registry() {
    static MSE_Loss mse_loss;
    static map<string, const TrainingLoss*> losses{
        {mse_loss.get_name(), &mse_loss}
    };
    return losses;
}
}  // namespace

const TrainingLoss* default_training_loss() {
    return get_training_loss("mse");
}

const TrainingLoss* get_training_loss(const string& name) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it = registry().find(name);
    return it == registry().end() ? nullptr : it->second;
}

bool register_training_loss(const TrainingLoss* loss) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it = registry().find(loss->get_name());
    if (it != registry().end()) {
        return it->second == loss;
    }
    registry()[loss->get_name()] = loss;
    return true;
}

vector<string> training_loss_names() {
    std::lock_guard<std::mutex> lock(registry_mutex);
    vector<string> names;
    for (auto& kv : registry()) {
        names.push_back(kv.first);
    }
    return names;
}
