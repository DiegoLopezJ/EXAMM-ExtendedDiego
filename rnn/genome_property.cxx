#include "rnn/genome_property.hxx"

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "rnn/evaluation_metric.hxx"
#include "rnn/training_loss.hxx"

GenomeProperty::GenomeProperty() {
    bp_iterations = 10;
    dropout_probability = 0.0;
    weight_decay = 0.0;
    min_recurrent_depth = 1;
    max_recurrent_depth = 10;

    training_loss = default_training_loss();
    checkpoint_metric = EvaluationMetric::MSE;
    fitness_metric = EvaluationMetric::MSE;
}

static EvaluationMetric evaluation_metric_argument(const vector<string>& arguments, const string& flag) {
    EvaluationMetric metric = EvaluationMetric::MSE;
    string value;
    if (get_argument(arguments, flag, false, value) && !parse_evaluation_metric(value, metric)) {
        Log::fatal("ERROR: unknown %s '%s', options are: mse, mae, return\n", flag.c_str(), value.c_str());
        exit(1);
    }
    return metric;
}

void GenomeProperty::generate_genome_property_from_arguments(const vector<string>& arguments) {
    get_argument(arguments, "--bp_iterations", true, bp_iterations);
    use_dropout = get_argument(arguments, "--dropout_probability", false, dropout_probability);
    get_argument(arguments, "--weight_decay", false, weight_decay);

    get_argument(arguments, "--min_recurrent_depth", false, min_recurrent_depth);
    get_argument(arguments, "--max_recurrent_depth", false, max_recurrent_depth);

    string training_loss_name = "mse";
    get_argument(arguments, "--training_loss", false, training_loss_name);
    training_loss = get_training_loss(training_loss_name);
    if (training_loss == nullptr) {
        string options;
        for (const string& name : training_loss_names()) {
            options += (options.empty() ? "" : ", ") + name;
        }
        Log::fatal(
            "ERROR: unknown --training_loss '%s', options are: %s\n", training_loss_name.c_str(), options.c_str()
        );
        exit(1);
    }
    checkpoint_metric = evaluation_metric_argument(arguments, "--checkpoint_metric");
    fitness_metric = evaluation_metric_argument(arguments, "--fitness_metric");

    // --objectives mse,return: Pareto search on these validation metrics; fitness (global best, extinction
    // ranking, logs) is then the first objective
    vector<string> objective_words;
    if (get_argument_vector(arguments, "--objectives", false, objective_words)) {
        vector<EvaluationMetric> objectives;
        if (!parse_evaluation_metric_list(objective_words, objectives) || objectives.size() < 2) {
            Log::fatal("ERROR: --objectives needs two or more different metrics from: mse, mae, return\n");
            exit(1);
        }
        if (argument_exists(arguments, "--fitness_metric") && fitness_metric != objectives[0]) {
            Log::fatal(
                "ERROR: --fitness_metric %s conflicts with --objectives (fitness is the first objective, %s)\n",
                evaluation_metric_name(fitness_metric).c_str(), evaluation_metric_name(objectives[0]).c_str()
            );
            exit(1);
        }
        fitness_metric = objectives[0];
        set_search_objectives(objectives);
        string names;
        for (EvaluationMetric m : objectives) {
            names += (names.empty() ? "" : ", ") + evaluation_metric_name(m);
        }
        Log::info(
            "Multi-objective search: islands keep genomes by Pareto rank on validation %s (NSGA-II)\n", names.c_str()
        );
    }

    Log::info("Each generated genome is trained for %d epochs\n", bp_iterations);
    Log::info(
        "Use dropout is set to %s, dropout probability is %f\n", use_dropout ? "True" : "False", dropout_probability
    );
    if (weight_decay > 0.0) {
        Log::info("L2 weight decay enabled: lambda=%.6f\n", weight_decay);
    } else {
        Log::info("L2 weight decay disabled (lambda=0.0)\n");
    }
    Log::info("Min recurrent depth is %d, max recurrent depth is %d\n", min_recurrent_depth, max_recurrent_depth);
    Log::info(
        "Training loss: %s, checkpoint selection: lowest validation %s, fitness: validation %s\n",
        training_loss->get_name().c_str(), evaluation_metric_name(checkpoint_metric).c_str(),
        evaluation_metric_name(fitness_metric).c_str()
    );
}

void GenomeProperty::set_genome_properties(RNN_Genome* genome) {
    genome->set_bp_iterations(bp_iterations);
    if (use_dropout) {
        genome->enable_dropout(dropout_probability);
    }
    genome->set_weight_decay(weight_decay);
    genome->set_training_loss(training_loss);
    genome->set_checkpoint_metric(checkpoint_metric);
    genome->set_fitness_metric(fitness_metric);
    genome->normalize_type = normalize_type;
    genome->set_parameter_names(input_parameter_names, output_parameter_names);
    genome->set_normalize_bounds(normalize_type, normalize_mins, normalize_maxs, normalize_avgs, normalize_std_devs);
}

void GenomeProperty::get_time_series_parameters(TimeSeriesSets* time_series_sets) {
    input_parameter_names = time_series_sets->get_input_parameter_names();
    output_parameter_names = time_series_sets->get_output_parameter_names();
    normalize_type = time_series_sets->get_normalize_type();
    normalize_mins = time_series_sets->get_normalize_mins();
    normalize_maxs = time_series_sets->get_normalize_maxs();
    normalize_avgs = time_series_sets->get_normalize_avgs();
    normalize_std_devs = time_series_sets->get_normalize_std_devs();
    number_inputs = time_series_sets->get_number_inputs();
    number_outputs = time_series_sets->get_number_outputs();
}

uniform_int_distribution<int32_t> GenomeProperty::get_recurrent_depth_dist() {
    return uniform_int_distribution<int32_t>(this->min_recurrent_depth, this->max_recurrent_depth);
}