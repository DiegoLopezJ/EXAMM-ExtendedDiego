// Tests for the multi-objective search (--objectives mse,return), ported from EXAMM-MO onto the evaluation policies:
//   1. dominance on (lower validation MSE, higher validation return); fitness = the first objective (MSE)
//   2. an island fed more genomes than it holds keeps every genome no other inserted genome dominates (NSGA-II)
//   3. untrained placeholders fill an island while it initializes and are the first removed
//   4. the global Pareto archive is exactly the non-dominated set; with a cap it keeps the ends of the front
//   5. pareto_front.csv: one column per objective, the return in its own units (higher is better)
//   6. single-objective search (no --objectives) is unchanged: islands rank by fitness alone
// Exits with status 1 if any check fails.

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <random>
#include <set>
#include <string>
#include <vector>
using std::string;
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "examm/island.hxx"
#include "examm/island_speciation_strategy.hxx"
#include "rnn/evaluation_metric.hxx"
#include "rnn/generate_nn.hxx"
#include "rnn/rnn_genome.hxx"
#include "weights/weight_rules.hxx"

static int32_t failures = 0, checks = 0;

static void check(bool condition, const string& what) {
    checks++;
    if (!condition) {
        failures++;
        Log::error("FAILED: %s\n", what.c_str());
    }
}

static WeightRules* weight_rules;
static std::mt19937 generator(11);

// structurally different genomes (an island keeps only one of identical structures), with set validation results
static RNN_Genome* make_genome(int32_t index, double mse, double validation_return) {
    int32_t layers = 1 + index % 2, nodes = 1 + index / 2;
    RNN_Genome* g = create_elman(vector<string>{"a", "b"}, layers, nodes, vector<string>{"RET"}, 1, weight_rules);
    g->initialize_randomly();
    vector<double> w;
    g->get_weights(w);
    g->set_initial_parameters(w);
    g->set_best_parameters(w);
    g->set_generation_id(index);
    g->set_group_id(0);
    g->set_fitness_metric(EvaluationMetric::MSE);
    g->set_validation_scores(mse, 2 * mse, validation_return);
    return g;
}

static bool contains(Island& island, int32_t generation_id) {
    for (RNN_Genome* g : island.get_genomes()) {
        if (g->get_generation_id() == generation_id) {
            return true;
        }
    }
    return false;
}

static vector<std::pair<double, double> > random_points(int32_t n) {
    std::uniform_real_distribution<double> mse(0.5, 1.5), ret(-0.3, 0.6);
    vector<std::pair<double, double> > p;
    for (int32_t i = 0; i < n; i++) {
        p.push_back({mse(generator), ret(generator)});
    }
    return p;
}

// indexes of the points no other point dominates (lower MSE, higher return)
static std::set<int32_t> non_dominated(const vector<std::pair<double, double> >& p) {
    std::set<int32_t> keep;
    for (int32_t i = 0; i < (int32_t) p.size(); i++) {
        bool dominated = false;
        for (int32_t j = 0; j < (int32_t) p.size() && !dominated; j++) {
            dominated = j != i && p[j].first <= p[i].first && p[j].second >= p[i].second
                        && (p[j].first < p[i].first || p[j].second > p[i].second);
        }
        if (!dominated) {
            keep.insert(i);
        }
    }
    return keep;
}

static void test_dominance() {
    RNN_Genome* a = make_genome(0, 0.8, 0.10);
    RNN_Genome* b = make_genome(1, 0.9, 0.05);  // worse on both
    RNN_Genome* c = make_genome(2, 0.7, 0.00);  // better MSE, worse return than a
    RNN_Genome* d = make_genome(3, 0.8, 0.10);  // equal to a
    check(a->dominates(b) && !b->dominates(a), "lower MSE and higher return dominates");
    check(!a->dominates(c) && !c->dominates(a), "a trade-off: neither dominates");
    check(!a->dominates(d) && !d->dominates(a), "equal genomes do not dominate each other");
    check(a->get_fitness() == 0.8, "fitness is the first objective (validation MSE)");
    vector<double> v = a->get_objective_values();
    check(v.size() == 2 && v[0] == 0.8 && v[1] == -0.10, "objective values: MSE, -return (both lower is better)");
    RNN_Genome* untrained = make_genome(4, EXAMM_MAX_DOUBLE, -EXAMM_MAX_DOUBLE);
    check(c->dominates(untrained), "every trained genome dominates an untrained placeholder");
    for (RNN_Genome* g : {a, b, c, d, untrained}) {
        delete g;
    }
}

static void test_island_keeps_front() {
    for (int32_t trial = 0; trial < 5; trial++) {
        vector<std::pair<double, double> > p = random_points(40);
        std::set<int32_t> front = non_dominated(p);
        if (front.size() > 10) {
            continue;  // the island cannot hold the whole front; not what this test is about
        }
        Island island(0, 10);
        for (int32_t i = 0; i < 40; i++) {
            RNN_Genome* g = make_genome(i, p[i].first, p[i].second);
            island.insert_genome(g);
            delete g;
        }
        std::set<int32_t> kept;
        for (int32_t i = 0; i < 40; i++) {
            if (contains(island, i)) {
                kept.insert(i);
            }
        }
        bool all = std::includes(kept.begin(), kept.end(), front.begin(), front.end());
        check(island.size() == 10, "island trial " + std::to_string(trial) + ": stays at its size (10)");
        check(all, "island trial " + std::to_string(trial) + ": every non-dominated genome of the 40 survives");
        check(
            island.get_best_fitness() == std::min_element(p.begin(), p.end())->first,
            "island trial " + std::to_string(trial) + ": best fitness = lowest MSE (extinction ranking)"
        );
    }
}

static void test_placeholders() {
    Island island(0, 3);
    for (int32_t i = 0; i < 3; i++) {
        RNN_Genome* g = make_genome(i, EXAMM_MAX_DOUBLE, -EXAMM_MAX_DOUBLE);
        check(island.insert_genome(g) >= 0, "placeholder " + std::to_string(i) + " is inserted while initializing");
        delete g;
    }
    check(island.is_full(), "placeholders fill the island");
    for (int32_t i = 3; i < 6; i++) {
        RNN_Genome* g = make_genome(i, 1.0 - 0.1 * i, 0.01 * i);
        island.insert_genome(g);
        delete g;
    }
    bool none_left = !contains(island, 0) && !contains(island, 1) && !contains(island, 2);
    check(
        none_left && contains(island, 3) && contains(island, 4) && contains(island, 5),
        "trained genomes replace every placeholder"
    );
}

static IslandSpeciationStrategy* make_strategy(RNN_Genome* seed) {
    IslandSpeciationStrategy* strategy = new IslandSpeciationStrategy(
        1, 50, 0.7, 0.3, 0.0, seed, "", "", 0, 1, 0, 1000, false, false, false, "", 0, false
    );
    std::function<void(int32_t, RNN_Genome*)> no_mutation = [](int32_t, RNN_Genome*) {};
    strategy->initialize_population(no_mutation);  // creates the islands, as EXAMM does before inserting
    return strategy;
}

static void test_archive(const string& directory) {
    vector<std::pair<double, double> > p = random_points(60);
    std::set<int32_t> front = non_dominated(p);

    RNN_Genome* seed = make_genome(999, EXAMM_MAX_DOUBLE, -EXAMM_MAX_DOUBLE);
    IslandSpeciationStrategy* strategy = make_strategy(seed);
    for (int32_t i = 0; i < 60; i++) {
        RNN_Genome* g = make_genome(i, p[i].first, p[i].second);
        strategy->insert_genome(g);
        delete g;
    }
    std::set<int32_t> archived;
    for (RNN_Genome* g : strategy->get_pareto_archive()) {
        archived.insert(g->get_generation_id());
    }
    check(archived == front, "archive = exactly the non-dominated genomes (" + std::to_string(front.size()) + ")");

    // pareto_front.csv
    strategy->save_pareto_archive(directory);
    std::ifstream in(directory + "/pareto_front.csv");
    string header, row;
    std::getline(in, header);
    check(
        header == "genome_id,val_mse,val_return,enabled_nodes,enabled_edges,enabled_recurrent_edges",
        "pareto_front.csv header names the objectives"
    );
    int32_t rows = 0;
    bool returns_ok = true;
    while (std::getline(in, row)) {
        int32_t id;
        double mse, ret;
        if (sscanf(row.c_str(), "%d,%lf,%lf", &id, &mse, &ret) == 3) {
            returns_ok = returns_ok && std::fabs(mse - p[id].first) < 1e-8 && std::fabs(ret - p[id].second) < 1e-8;
            rows++;
        }
    }
    check(rows == (int32_t) front.size() && returns_ok, "pareto_front.csv: one row per genome, MSE and net return");
    delete strategy;

    // with a cap: never more than the cap, all non-dominated, the two ends kept
    if (front.size() > 3) {
        IslandSpeciationStrategy* capped = make_strategy(seed);
        capped->set_pareto_archive_size(3);
        for (int32_t i = 0; i < 60; i++) {
            RNN_Genome* g = make_genome(i, p[i].first, p[i].second);
            capped->insert_genome(g);
            delete g;
        }
        std::set<int32_t> c;
        for (RNN_Genome* g : capped->get_pareto_archive()) {
            c.insert(g->get_generation_id());
        }
        int32_t lowest_mse = *std::min_element(front.begin(), front.end(), [&](int32_t a, int32_t b) {
            return p[a].first < p[b].first;
        });
        int32_t highest_return = *std::max_element(front.begin(), front.end(), [&](int32_t a, int32_t b) {
            return p[a].second < p[b].second;
        });
        check(c.size() == 3, "capped archive holds 3 genomes");
        check(std::includes(front.begin(), front.end(), c.begin(), c.end()), "capped archive: all non-dominated");
        check(c.count(lowest_mse) && c.count(highest_return), "capped archive keeps both ends of the front");
        delete capped;
    }
    delete seed;
}

static void test_single_objective_unchanged() {
    set_search_objectives({});
    check(!is_multi_objective_search(), "no --objectives: single-objective search");
    Island island(0, 3);
    // a genome with a terrible return but the lowest MSE must rank first, as in the original EXAMM
    vector<std::pair<double, double> > p{
        {0.9,  0.5},
        {0.5, -0.9},
        {0.7,  0.0},
        {0.6,  0.1}
    };
    for (int32_t i = 0; i < 4; i++) {
        RNN_Genome* g = make_genome(i, p[i].first, p[i].second);
        island.insert_genome(g);
        delete g;
    }
    check(
        island.get_best_fitness() == 0.5 && contains(island, 1) && !contains(island, 0),
        "single objective: ranked by MSE only, the highest-MSE genome dropped"
    );
}

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);
    string directory = "./test_pareto_search_output";
    if (!argument_exists(arguments, "--output_directory")) {
        arguments.push_back("--output_directory");
        arguments.push_back(directory);
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
    weight_rules = new WeightRules();
    weight_rules->initialize_from_args(arguments);

    vector<EvaluationMetric> objectives;
    check(
        parse_evaluation_metric_list({"mse,return"}, objectives) && objectives.size() == 2
            && objectives[0] == EvaluationMetric::MSE && objectives[1] == EvaluationMetric::RETURN,
        "--objectives mse,return parses"
    );
    vector<EvaluationMetric> words;
    check(
        parse_evaluation_metric_list({"mse", "return"}, words) && words == objectives,
        "--objectives mse return parses the same"
    );
    vector<EvaluationMetric> bad;
    check(
        !parse_evaluation_metric_list({"mse,mse"}, bad) && !parse_evaluation_metric_list({"mse,rmse"}, bad),
        "repeated or unknown objectives are rejected"
    );
    set_search_objectives(objectives);

    test_dominance();
    test_island_keeps_front();
    test_placeholders();
    test_archive(directory);
    test_single_objective_unchanged();

    delete weight_rules;
    printf("%d of %d checks passed\n", checks - failures, checks);
    if (failures > 0) {
        printf("SOME FAILED\n");
        return 1;
    }
    printf("ALL PASSED\n");
    return 0;
}
