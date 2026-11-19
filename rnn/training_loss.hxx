#ifndef EXAMM_TRAINING_LOSS_HXX
#define EXAMM_TRAINING_LOSS_HXX

#include <cstdint>
#include <string>
using std::string;

#include <vector>
using std::vector;

/**
 * The objective backpropagation minimizes (the training loss). It is separate from the metric that picks which
 * weights to keep (checkpoint selection) and from the metric evolution ranks genomes by (fitness); see
 * evaluation_metric.hxx.
 *
 * For one series, compute() returns the loss L and writes one value e[i][j] per output i and time step j.
 * Backpropagation then uses
 *
 *     e[i][j] * backward_scale(L, T)                     (T = series length)
 *
 * as the derivative of the training objective J with respect to output i at time j, where J is defined by
 *
 *     dJ/dw = gradient_check_scale(L) * dL/dw.
 *
 * The numerical gradient check (RNN::get_empirical_gradient) computes the right-hand side from finite differences of
 * L, so a loss is correct when its analytic and numerical gradients agree. A plain loss returns the full derivative
 * dL/dy in e and leaves both scales at 1. MSE_Loss keeps EXAMM's established scaling: e = y - t and
 * backward_scale = L * 2 / T, which is the gradient of L^2 / 2 (gradient_check_scale = L).
 */
class TrainingLoss {
   public:
    virtual ~TrainingLoss() = default;

    // the name used on the command line and in serialized genomes; must be unique among registered losses
    virtual string get_name() const = 0;

    virtual double compute(
        const vector<vector<double> >& predicted, const vector<vector<double> >& expected,
        vector<vector<double> >& output_errors
    ) const = 0;

    virtual double backward_scale(double loss, int32_t series_length) const {
        (void) loss;
        (void) series_length;
        return 1.0;
    }

    virtual double gradient_check_scale(double loss) const {
        (void) loss;
        return 1.0;
    }
};

// EXAMM's original objective: the sum over outputs of each output's mean squared error, with the original gradient
// scaling. The default everywhere.
class MSE_Loss : public TrainingLoss {
   public:
    string get_name() const override;
    double compute(
        const vector<vector<double> >& predicted, const vector<vector<double> >& expected,
        vector<vector<double> >& output_errors
    ) const override;
    double backward_scale(double loss, int32_t series_length) const override;
    double gradient_check_scale(double loss) const override;
};

// The loss genomes train with unless told otherwise (MSE_Loss).
const TrainingLoss* default_training_loss();

// Looks a loss up by name; nullptr if no loss of that name is registered. "mse" is always registered.
const TrainingLoss* get_training_loss(const string& name);

// Makes a custom loss available by name (for deserializing genomes that use it). The loss is not owned and must
// outlive every genome that uses it. Returns false if a different loss already has that name.
bool register_training_loss(const TrainingLoss* loss);

// The names accepted by --training_loss.
vector<string> training_loss_names();

#endif
