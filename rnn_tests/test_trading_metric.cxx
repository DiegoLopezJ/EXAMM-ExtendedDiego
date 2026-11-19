// Tests for the validation trading-return metric (rnn/trading_metric.hxx) and EvaluationMetric::RETURN:
//   1. paper_long_short on a hand-worked case (long the top forecast, short the bottom, costs on both legs)
//   2. costs: a flat price loses the spread on both sides, longs and shorts alike; costs off -> 0
//   3. the sign check: no trade when the k-th highest forecast is not > 0 or the k-th lowest not < 0
//   4. ties are broken by stock name (as the Python backtest's stable sort over sorted names)
//   5. the genome: checkpoint by return keeps the highest-return weights; fitness = -return; the return survives
//      serialization; files with the earlier policy block ("EVL1") and EXAMM-MO's "MOBJ" block still load
// The exact agreement with the Python backtest on real genomes is checked by analysis/check_cpp_trading_metric.py.
// Exits with status 1 if any check fails.

#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cstring>
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
#include "rnn/trading_metric.hxx"
#include "weights/weight_rules.hxx"

typedef vector<vector<double> > Matrix;

static int32_t failures = 0, checks = 0;

static void check(bool condition, const string& what) {
    checks++;
    if (!condition) {
        failures++;
        Log::error("FAILED: %s\n", what.c_str());
    }
}

static bool close(double a, double b, double tol = 1e-12) {
    return std::fabs(a - b) <= tol * std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
}

static Matrix constant(int32_t n_stocks, int32_t n_days, double value) {
    return Matrix(n_stocks, vector<double>(n_days, value));
}

static void test_hand_worked() {
    // 4 stocks, k = 1, 3 days. Day 0: long A (forecast +2 %), short D (-2 %). Day 1: every forecast positive, so the
    // sign check keeps the book. Day 2: close. Prices 100 on days 0-1; day 2: A 110, D 90. Cost 1 per share.
    vector<string> names{"A", "B", "C", "D"};
    Matrix f{
        { 0.02,  0.03, 0.0},
        { 0.01,  0.02, 0.0},
        {-0.01,  0.01, 0.0},
        {-0.02, 0.005, 0.0}
    };
    Matrix p = constant(4, 3, 100.0);
    p[0][2] = 110.0;
    p[3][2] = 90.0;
    Matrix c = constant(4, 3, 1.0);
    TradingResult r = paper_long_short(f, p, c, 1, true, names);

    double shares = 1000.0 / 101.0;     // each leg: 1000 / (price + cost)
    double cash = 0.0 + shares * 99.0;  // long paid 1000; short sale credited at price - cost
    cash += (110.0 - 1.0) * shares;     // sell A at price - cost
    cash += (90.0 + 1.0) * -shares;     // cover D at price + cost
    check(close(r.total_return, cash / 1000.0 - 1.0), "hand-worked long/short return");
    check(r.days_traded == 1 && r.days == 3, "hand-worked: one trading day, the sign check skipped day 1");
}

static void test_costs() {
    // flat prices; one long and one short held one day: each leg pays the spread twice
    vector<string> names{"A", "B"};
    Matrix f{
        { 0.01, 0.0},
        {-0.01, 0.0}
    };
    Matrix p = constant(2, 2, 100.0);
    Matrix c = constant(2, 2, 1.0);
    TradingResult r = paper_long_short(f, p, c, 1, true, names);
    double shares = 1000.0 / 101.0;
    double expected = (shares * 99.0 + (100.0 - 1.0) * shares + (100.0 + 1.0) * -shares) / 1000.0 - 1.0;
    check(close(r.total_return, expected), "flat price: costs on both legs");
    check(r.total_return < 0, "flat price with costs loses money");

    // long-only loss == short-only loss on a flat price: long A / short B vs long B / short A give the same result
    Matrix g{
        {-0.01, 0.0},
        { 0.01, 0.0}
    };
    check(
        close(paper_long_short(g, p, c, 1, true, names).total_return, r.total_return),
        "flat price: swapping the long and short side costs the same"
    );

    Matrix zero = constant(2, 2, 0.0);
    check(paper_long_short(f, p, zero, 1, true, names).total_return == 0.0, "flat price, no costs: return 0");

    // a 10 % move in the long's favour without costs: +10 % on the long leg's capital
    Matrix up = constant(2, 2, 100.0);
    up[0][1] = 110.0;
    check(close(paper_long_short(f, up, zero, 1, true, names).total_return, 0.10), "no costs: long +10 %");
}

static void test_sign_check() {
    vector<string> names{"A", "B", "C", "D"};
    Matrix f = constant(4, 5, 0.01);  // every forecast positive: never trades with the sign check
    for (int32_t i = 0; i < 4; i++) {
        f[i][0] = 0.01 * (i + 1);
    }
    Matrix p = constant(4, 5, 100.0);
    p[0][4] = 50.0;
    Matrix c = constant(4, 5, 0.5);
    TradingResult r = paper_long_short(f, p, c, 1, true, names);
    check(r.days_traded == 0 && r.total_return == 0.0, "sign check: no trade when no forecast is negative");
    TradingResult every_day = paper_long_short(f, p, c, 1, false, names);
    check(every_day.days_traded == 4, "without the sign check it trades every day but the last");
}

static void test_ties_by_name() {
    // equal forecasts: the Python backtest's stable argsort over name-sorted stocks puts the first name lowest, so
    // the long is the last name and the short the first name
    vector<string> names{"C", "A", "B", "D"};  // series order differs from name order
    Matrix f{
        { 0.01, 0},
        { 0.01, 0},
        {-0.01, 0},
        {-0.01, 0}
    };  // C, A tied high; B, D tied low
    Matrix p = constant(4, 2, 100.0);
    p[0][1] = 120.0;  // C
    p[1][1] = 80.0;   // A
    p[2][1] = 80.0;   // B
    p[3][1] = 120.0;  // D
    Matrix zero = constant(4, 2, 0.0);
    // ascending by forecast, ties in name order: B(-), D(-), A(+), C(+) -> short B, long C
    TradingResult r = paper_long_short(f, p, zero, 1, true, names);
    // long C: 10 shares sold at 120; short B: 10 shares credited 1000, covered at 80
    double expected = (1000.0 + 10.0 * 120.0 - 10.0 * 80.0) / 1000.0 - 1.0;
    check(close(r.total_return, expected), "ties broken by stock name: long C, short B");
}

static Matrix random_matrix(std::mt19937& g, int32_t rows, int32_t cols, double lo, double hi) {
    std::uniform_real_distribution<double> u(lo, hi);
    Matrix m(rows, vector<double>(cols));
    for (auto& r : m) {
        for (auto& v : r) {
            v = u(g);
        }
    }
    return m;
}

static RNN_Genome* round_trip(RNN_Genome* genome) {
    char* bytes;
    int32_t length;
    genome->write_to_array(&bytes, length);
    RNN_Genome* copy = new RNN_Genome(bytes, length);
    free(bytes);
    return copy;
}

static void test_genome(WeightRules* weight_rules) {
    std::mt19937 g(7);
    const int32_t n_stocks = 6, n_days = 40;
    vector<string> stock_names;
    for (int32_t i = 0; i < n_stocks; i++) {
        stock_names.push_back("S" + std::to_string(i));
    }
    // validation series: one per stock, 2 inputs, 1 output (the return forecast)
    vector<vector<vector<double> > > vin, vout;
    for (int32_t i = 0; i < n_stocks; i++) {
        vin.push_back(random_matrix(g, 2, n_days, -1, 1));
        vout.push_back(random_matrix(g, 1, n_days, -1, 1));
    }
    TradingData data;
    data.stocks = stock_names;
    data.prices = random_matrix(g, n_stocks, n_days + 1, 50, 150);
    data.costs = random_matrix(g, n_stocks, n_days + 1, 0.01, 0.1);
    data.ret_median = 0.0;
    data.ret_iqr = 0.02;
    set_validation_trading_data(data, 2);

    RNN_Genome* genome = create_elman(vector<string>{"a", "b"}, 1, 3, vector<string>{"RET"}, 2, weight_rules);
    genome->set_stochastic(false);
    int32_t n = genome->get_number_weights();

    // the trading return equals paper_long_short on the genome's forecasts
    vector<double> w = random_matrix(g, 1, n, -0.5, 0.5)[0];
    vector<vector<double> > pred = genome->get_predictions(w, vin, vout);
    Matrix raw(n_stocks);
    for (int32_t i = 0; i < n_stocks; i++) {
        for (double v : pred[i]) {
            raw[i].push_back(genome->denormalize_output(0, v) * data.ret_iqr + data.ret_median);
        }
    }
    double direct = paper_long_short(raw, data.prices, data.costs, 2, true, stock_names).total_return;
    check(
        genome->get_trading_return(w, vin, vout) == direct, "genome trading return = paper_long_short on its forecasts"
    );

    // candidates: A has the lower validation MSE, B the higher return
    vector<vector<double> > cand;
    vector<double> mse, ret;
    for (int32_t i = 0; i < 300; i++) {
        cand.push_back(random_matrix(g, 1, n, -1, 1)[0]);
        mse.push_back(genome->get_mse(cand.back(), vin, vout));
        ret.push_back(genome->get_trading_return(cand.back(), vin, vout));
    }
    int32_t a = -1, b = -1;
    for (int32_t i = 0; i < 300 && a < 0; i++) {
        for (int32_t j = 0; j < 300; j++) {
            if (mse[i] < mse[j] && ret[i] < ret[j]) {
                a = i;
                b = j;
                break;
            }
        }
    }
    check(a >= 0, "found weights ranked differently by MSE and return");
    if (a < 0) {
        delete genome;
        return;
    }

    for (EvaluationMetric checkpoint : {EvaluationMetric::MSE, EvaluationMetric::RETURN}) {
        string tag = "checkpoint " + evaluation_metric_name(checkpoint);
        RNN_Genome* x = genome->copy();
        x->set_checkpoint_metric(checkpoint);
        x->set_fitness_metric(EvaluationMetric::RETURN);
        x->update_checkpoint(cand[a], mse[a], vin, vout);
        x->update_checkpoint(cand[b], mse[b], vin, vout);
        int32_t kept = checkpoint == EvaluationMetric::RETURN ? b : a;
        check(x->get_best_parameters() == cand[kept], tag + ": kept weights follow the checkpoint metric");
        x->finish_validation_return(vin, vout);
        check(x->get_best_validation_return() == ret[kept], tag + ": recorded return is that of the kept weights");
        check(x->get_best_validation_mse() == mse[kept], tag + ": recorded MSE is that of the kept weights");
        check(x->get_fitness() == -ret[kept], tag + ": return fitness = -validation return (lower is better)");

        RNN_Genome* y = round_trip(x);
        check(
            y->get_best_validation_return() == x->get_best_validation_return() && y->get_fitness() == x->get_fitness()
                && y->get_checkpoint_metric() == checkpoint && y->get_fitness_metric() == EvaluationMetric::RETURN,
            tag + ": return, metrics and fitness survive serialization (MPI path)"
        );
        delete y;
        delete x;
    }

    // older blocks: "EVL1" (no return) and EXAMM-MO's "MOBJ" in front of it
    RNN_Genome* x = genome->copy();
    x->set_fitness_metric(EvaluationMetric::MAE);
    char* bytes;
    int32_t length;
    x->write_to_array(&bytes, length);
    string s(bytes, length);
    free(bytes);
    int32_t block = 4 + 4 + 3 + 4 + 4 + 8;  // "EVL2", "mse", checkpoint, fitness, return
    string body = s.substr(0, s.size() - block);
    string evl1 = s.substr(s.size() - block, block - 8);
    int32_t v1 = 0x45564C31;
    memcpy(&evl1[0], &v1, 4);

    string old1 = body + evl1;
    RNN_Genome* y = new RNN_Genome(&old1[0], (int32_t) old1.size());
    check(
        y->get_fitness_metric() == EvaluationMetric::MAE && y->get_best_validation_return() == -EXAMM_MAX_DOUBLE,
        "an EVL1 genome loads with its metrics and no return"
    );
    delete y;

    double mo_value = 1.25;
    string mobj = string("MOBJ", 4) + string((char*) &mo_value, 8);
    string mo = body + mobj;
    y = new RNN_Genome(&mo[0], (int32_t) mo.size());
    check(
        y->get_fitness_metric() == EvaluationMetric::MSE && y->get_best_parameters() == x->get_best_parameters(),
        "an EXAMM-MO genome (MOBJ block) loads as an ordinary MSE genome"
    );
    delete y;
    string mo_then_policy = body + mobj + s.substr(s.size() - block);
    y = new RNN_Genome(&mo_then_policy[0], (int32_t) mo_then_policy.size());
    check(y->get_fitness_metric() == EvaluationMetric::MAE, "an MOBJ block followed by a policy block reads both");
    delete y;
    delete x;
    delete genome;
}

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);
    string directory = "./test_trading_metric_output";
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

    WeightRules* weight_rules = new WeightRules();
    weight_rules->initialize_from_args(arguments);

    test_hand_worked();
    test_costs();
    test_sign_check();
    test_ties_by_name();
    test_genome(weight_rules);

    delete weight_rules;
    printf("%d of %d checks passed\n", checks - failures, checks);
    if (failures > 0) {
        printf("SOME FAILED\n");
        return 1;
    }
    printf("ALL PASSED\n");
    return 0;
}
