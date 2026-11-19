#ifndef RNN_BPTT_HXX
#define RNN_BPTT_HXX

#include <fstream>
using std::ifstream;
using std::istream;
using std::ofstream;
using std::ostream;

#include <map>
using std::map;

#include <random>
using std::minstd_rand0;
using std::mt19937;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

#include <vector>
using std::vector;

#include "common/random.hxx"
#include "evaluation_metric.hxx"
#include "rnn.hxx"
#include "rnn_edge.hxx"
#include "rnn_node_interface.hxx"
#include "rnn_recurrent_edge.hxx"
#include "time_series/time_series.hxx"
#include "trading_metric.hxx"
#include "training_loss.hxx"
#include "weights/weight_rules.hxx"
#include "weights/weight_update.hxx"
// #include "word_series/word_series.hxx"

// mysql can't handle the max float value for some reason
#define EXAMM_MAX_DOUBLE 10000000

extern vector<int32_t> dnas_node_types;

string parse_fitness(double fitness);

class RNN_Genome {
   private:
    int32_t generation_id;
    int32_t group_id;

    int32_t bp_iterations;

    bool use_dropout;
    double dropout_probability;

    double weight_decay;  // L2 weight decay coefficient (0.0 = disabled)

    string structural_hash;

    string log_filename;

    WeightRules* weight_rules;

    map<string, int32_t> generated_by_map;

    vector<double> initial_parameters;

    double best_validation_mse;
    double best_validation_mae;
    // validation net return of the trading rule at best_parameters (trading_metric.hxx); -EXAMM_MAX_DOUBLE when not
    // computed (no --trading_file), NaN when undefined
    double best_validation_return;
    vector<double> best_parameters;
    // multi-objective search: Pareto front + crowding term set by the island (lower is better); not serialized
    double selection_score;

    // Three independent decisions (defaults reproduce the original MSE-only behavior):
    const TrainingLoss* training_loss;   // what backpropagation minimizes (not owned)
    EvaluationMetric checkpoint_metric;  // which epoch's weights are kept as best_parameters
    EvaluationMetric fitness_metric;     // what get_fitness() returns, i.e. how evolution ranks the genome

    void read_evaluation_policies(istream& bin_istream);
    void write_evaluation_policies(ostream& bin_ostream);

    minstd_rand0 generator;

    uniform_real_distribution<double> rng;
    uniform_real_distribution<double> rng_0_1;
    uniform_real_distribution<double> rng_1_1;
    uniform_int_distribution<int32_t> rng_int;
    NormalDistribution normal_distribution;

    vector<RNN_Node_Interface*> nodes;
    vector<RNN_Edge*> edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    vector<string> input_parameter_names;
    vector<string> output_parameter_names;

    string normalize_type;
    map<string, double> normalize_mins;
    map<string, double> normalize_maxs;
    map<string, double> normalize_avgs;
    map<string, double> normalize_std_devs;

   public:
    void sort_nodes_by_depth();
    void sort_edges_by_depth();
    void sort_recurrent_edges_by_depth();

    RNN_Genome(
        vector<RNN_Node_Interface*>& _nodes, vector<RNN_Edge*>& _edges, vector<RNN_Recurrent_Edge*>& _recurrent_edges,
        WeightRules* weight_rules
    );
    RNN_Genome(
        vector<RNN_Node_Interface*>& _nodes, vector<RNN_Edge*>& _edges, vector<RNN_Recurrent_Edge*>& _recurrent_edges,
        int16_t seed, WeightRules* weight_rules
    );

    RNN_Genome* copy();

    ~RNN_Genome();

    static string print_statistics_header();
    string print_statistics();

    static RNN_Node_Interface* read_node_from_stream(istream& bin_istream);

    void set_parameter_names(
        const vector<string>& _input_parameter_names, const vector<string>& _output_parameter_names
    );

    string generated_by_string();

    string get_edge_count_str(bool recurrent);
    string get_node_count_str(int32_t node_type);

    const map<string, int32_t>* get_generated_by_map();

    double get_avg_recurrent_depth() const;

    int32_t get_enabled_edge_count();
    int32_t get_enabled_recurrent_edge_count();
    int32_t get_enabled_node_count(int32_t node_type);
    int32_t get_node_count(int32_t node_type);
    int32_t get_enabled_node_count();
    int32_t get_node_count();

    double get_fitness() const;

    void set_training_loss(const TrainingLoss* _training_loss);
    const TrainingLoss* get_training_loss() const;
    void set_checkpoint_metric(EvaluationMetric metric);
    EvaluationMetric get_checkpoint_metric() const;
    void set_fitness_metric(EvaluationMetric metric);
    EvaluationMetric get_fitness_metric() const;

    // Checkpoint selection: keeps parameters as best_parameters if they improve the checkpoint metric on the
    // validation data, recording both validation MSE and MAE of the kept weights. validation_mse is the validation
    // MSE of parameters (already computed by the caller). Returns true if the parameters were kept.
    bool update_checkpoint(
        const vector<double>& parameters, double validation_mse,
        const vector<vector<vector<double> > >& validation_inputs,
        const vector<vector<vector<double> > >& validation_outputs
    );
    double get_best_validation_softmax() const;
    double get_best_validation_mse() const;
    double get_best_validation_mae() const;
    double get_best_validation_return() const;

    // RET forecasts of every validation series turned into raw-return forecasts and traded with the paper's
    // long-short rule (trading_metric.hxx, the process-wide validation trading data); the net return, e.g.
    // 0.05 = +5 %. NaN if a forecast is not finite; exits if no trading data are configured.
    double get_trading_return(
        const vector<double>& parameters, const vector<vector<vector<double> > >& inputs,
        const vector<vector<vector<double> > >& outputs
    );
    // the value of one validation metric, oriented so that lower is better (MSE, MAE, -return); unset metrics are
    // EXAMM_MAX_DOUBLE (worst)
    double get_objective_value(EvaluationMetric metric) const;
    // the values of the search objectives (get_search_objectives()), lower is better
    vector<double> get_objective_values() const;
    // multi-objective search: no worse than other on every search objective and better on at least one
    bool dominates(const RNN_Genome* other) const;
    double get_selection_score() const;
    void set_selection_score(double score);
    // sets the recorded validation results of the kept weights (for tests and tools that score genomes elsewhere)
    void set_validation_scores(double mse, double mae, double validation_return);

    // after training: the validation return of the kept weights (if trading data are configured), unless the
    // checkpoint already tracked it
    void finish_validation_return(
        const vector<vector<vector<double> > >& validation_inputs,
        const vector<vector<vector<double> > >& validation_outputs
    );
    // the inverse of the time series normalization, for output parameter output_index
    double denormalize_output(int32_t output_index, double value) const;

    void set_normalize_bounds(
        string _normalize_type, const map<string, double>& _normalize_mins, const map<string, double>& _normalize_maxs,
        const map<string, double>& _normalize_avgs, const map<string, double>& _normalize_std_devs
    );

    string get_normalize_type() const;
    map<string, double> get_normalize_mins() const;
    map<string, double> get_normalize_maxs() const;
    map<string, double> get_normalize_avgs() const;
    map<string, double> get_normalize_std_devs() const;

    vector<string> get_input_parameter_names() const;
    vector<string> get_output_parameter_names() const;

    int32_t get_group_id() const;
    void set_group_id(int32_t _group_id);

    void set_bp_iterations(int32_t _bp_iterations);
    int32_t get_bp_iterations();

    void set_weight_decay(double _weight_decay);
    double get_weight_decay() const;

    // Turns on / off stochastic operations. If it is off, any stochastic values will be "frozen" in place.
    void set_stochastic(bool stochastic);
    void disable_dropout();
    void enable_dropout(double _dropout_probability);
    void set_log_filename(string _log_filename);

    void get_weights(vector<double>& parameters);
    void set_weights(const vector<double>& parameters);

    int32_t get_number_weights();
    int32_t get_number_inputs();
    int32_t get_number_outputs();

    double get_avg_edge_weight();
    void initialize_randomly();
    void initialize_xavier(RNN_Node_Interface* n);
    void initialize_kaiming(RNN_Node_Interface* n);
    void initialize_gp(RNN_Node_Interface* n);
    void initialize_node_randomly(RNN_Node_Interface* n);
    double get_xavier_weight(RNN_Node_Interface* output_node);
    double get_kaiming_weight(RNN_Node_Interface* output_node);
    double get_random_weight();

    void get_input_edges(
        int32_t node_innovation, vector<RNN_Edge*>& input_edges, vector<RNN_Recurrent_Edge*>& input_recurrent_edges
    );
    int32_t get_fan_in(int32_t node_innovation);
    int32_t get_fan_out(int32_t node_innovation);

    int32_t get_generation_id() const;
    void set_generation_id(int32_t generation_id);

    void clear_generated_by();
    void update_generation_map(map<string, int32_t>& generation_map);
    void set_generated_by(string type);
    int32_t get_generated_by(string type);

    RNN* get_rnn();
    vector<double> get_best_parameters() const;
    vector<double> get_initial_parameters() const;
    void set_best_parameters(vector<double> parameters);     // INFO: ADDED BY ABDELRAHMAN TO USE FOR TRANSFER LEARNING
    void set_initial_parameters(vector<double> parameters);  // INFO: ADDED BY ABDELRAHMAN TO USE FOR TRANSFER LEARNING
    void apply_homeostasis(double downscale_factor);          // SHY: multiplicative weight downscaling

    void get_analytic_gradient(
        vector<RNN*>& rnns, const vector<double>& parameters, const vector<vector<vector<double> > >& inputs,
        const vector<vector<vector<double> > >& outputs, double& mse, vector<double>& analytic_gradient, bool training
    );

    void backpropagate(
        const vector<vector<vector<double> > >& inputs, const vector<vector<vector<double> > >& outputs,
        const vector<vector<vector<double> > >& validation_inputs,
        const vector<vector<vector<double> > >& validation_outputs, WeightUpdate* weight_update_method
    );

    void backpropagate_stochastic(
        const vector<vector<vector<double> > >& inputs, const vector<vector<vector<double> > >& outputs,
        const vector<vector<vector<double> > >& validation_inputs,
        const vector<vector<vector<double> > >& validation_outputs, WeightUpdate* weight_update_method
    );

    double get_softmax(
        const vector<double>& parameters, const vector<vector<vector<double> > >& inputs,
        const vector<vector<vector<double> > >& outputs
    );
    double get_mse(
        const vector<double>& parameters, const vector<vector<vector<double> > >& inputs,
        const vector<vector<vector<double> > >& outputs
    );
    double get_mae(
        const vector<double>& parameters, const vector<vector<vector<double> > >& inputs,
        const vector<vector<vector<double> > >& outputs
    );

    vector<vector<double> > get_predictions(
        const vector<double>& parameters, const vector<vector<vector<double> > >& inputs,
        const vector<vector<vector<double> > >& outputs
    );
    void write_predictions(
        string output_directory, const vector<string>& input_filenames, const vector<double>& parameters,
        const vector<vector<vector<double> > >& inputs, const vector<vector<vector<double> > >& outputs,
        TimeSeriesSets* time_series_sets
    );
    // void write_predictions(string output_directory, const vector<string> &input_filenames, const vector<double>
    // &parameters, const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs,
    // Corpus * word_series_sets);

    void get_mu_sigma(const vector<double>& p, double& mu, double& sigma);

    bool sanity_check();
    void assign_reachability();
    bool outputs_unreachable();

    RNN_Node_Interface* create_node(
        double mu, double sigma, int32_t node_type, int32_t& node_innovation_count, double depth
    );

    bool attempt_edge_insert(
        RNN_Node_Interface* n1, RNN_Node_Interface* n2, double mu, double sigma, int32_t& edge_innovation_count
    );
    bool attempt_recurrent_edge_insert(
        RNN_Node_Interface* n1, RNN_Node_Interface* n2, double mu, double sigma, uniform_int_distribution<int32_t> dist,
        int32_t& edge_innovation_count
    );

    // after adding an Elman or Jordan node, generate the circular RNN edge for Elman and the
    // edges from output to this node for Jordan.
    void generate_recurrent_edges(
        RNN_Node_Interface* node, double mu, double sigma, uniform_int_distribution<int32_t> dist,
        int32_t& edge_innovation_count
    );

    bool add_edge(double mu, double sigma, int32_t& edge_innovation_count);
    bool add_recurrent_edge(
        double mu, double sigma, uniform_int_distribution<int32_t> rec_depth_dist, int32_t& edge_innovation_count
    );
    bool disable_edge();
    bool enable_edge();
    bool split_edge(
        double mu, double sigma, int32_t node_type, uniform_int_distribution<int32_t> rec_depth_dist,
        int32_t& edge_innovation_count, int32_t& node_innovation_count
    );

    bool add_node(
        double mu, double sigma, int32_t node_type, uniform_int_distribution<int32_t> dist,
        int32_t& edge_innovation_count, int32_t& node_innovation_count
    );

    bool enable_node();
    bool disable_node();
    bool split_node(
        double mu, double sigma, int32_t node_type, uniform_int_distribution<int32_t> dist,
        int32_t& edge_innovation_count, int32_t& node_innovation_count
    );
    bool merge_node(
        double mu, double sigma, int32_t node_type, uniform_int_distribution<int32_t> dist,
        int32_t& edge_innovation_count, int32_t& node_innovation_count
    );

    /**
     * Determines if the genome contains a node with the given innovation number
     *
     * @param the innovation number to fine
     *
     * @return true if the genome has a node with the provided innovation number, false otherwise.
     */
    bool has_node_with_innovation(int32_t innovation_number) const;

    bool equals(RNN_Genome* other);

    string get_color(double weight, bool is_recurrent);
    void write_graphviz(string filename);
    void print_equations();
    void write_equations(ostream& outstream);

    RNN_Genome(string binary_filename);
    RNN_Genome(char* array, int32_t length);
    RNN_Genome(istream& bin_infile);

    void read_from_array(char* array, int32_t length);
    void read_from_stream(istream& bin_istream);

    void write_to_array(char** array, int32_t& length);
    void write_to_file(string bin_filename);
    void write_to_stream(ostream& bin_stream);

    bool connect_new_input_node(
        double mu, double sig, RNN_Node_Interface* new_node, uniform_int_distribution<int32_t> dist,
        int32_t& edge_innovation_count, bool not_all_hidden
    );
    bool connect_new_output_node(
        double mu, double sig, RNN_Node_Interface* new_node, uniform_int_distribution<int32_t> dist,
        int32_t& edge_innovation_count, bool not_all_hidden
    );
    bool connect_node_to_hid_nodes(
        double mu, double sig, RNN_Node_Interface* new_node, uniform_int_distribution<int32_t> dist,
        int32_t& edge_innovation_count, bool from_input
    );
    vector<RNN_Node_Interface*> pick_possible_nodes(int32_t layer_type, bool not_all_hidden, string node_type);

    void update_innovation_counts(int32_t& node_innovation_count, int32_t& edge_innovation_count);

    vector<int32_t> get_innovation_list();
    /**
     * \return the structural hash (calculated when assign_reachaability is called)
     */
    string get_structural_hash() const;

    /**
     * \return the max innovation number of any node in the genome.
     */
    int32_t get_max_node_innovation_count();

    /**
     * \return the max innovation number of any edge or recurrent edge in the genome.
     */
    int32_t get_max_edge_innovation_count();

    ofstream* create_log_file();
    void update_log_file(
        ofstream* output_log, int32_t iteration, long milliseconds, double training_mse, double validation_mse,
        double avg_norm
    );

    void transfer_to(
        const vector<string>& new_input_parameter_names, const vector<string>& new_output_parameter_names,
        string transfer_learning_version, bool epigenetic_weights, int32_t min_recurrent_depth,
        int32_t max_recurrent_depth
    );

    friend class EXAMM;
    friend class IslandSpeciationStrategy;
    friend class NeatSpeciationStrategy;
    friend class RecDepthFrequencyTable;
    friend class GenomeProperty;
};

struct sort_genomes_by_fitness {
    bool operator()(RNN_Genome* g1, RNN_Genome* g2) {
        return g1->get_fitness() < g2->get_fitness();
    }
};

void write_binary_string(ostream& out, string s, string name);
void read_binary_string(istream& in, string& s, string name);

#endif
