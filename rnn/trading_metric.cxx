#include "rnn/trading_metric.hxx"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
using std::map;

#include "common/arguments.hxx"
#include "common/log.hxx"

namespace {
// position of one stock, as stock.py's Stock (share > 0 long, < 0 short)
double clear_holdings(double& share, double price, double cost) {
    double cash = 0.0;
    if (share < 0) {
        cash = (price + cost) * share;  // return_stock: covering a short is a buy
        share = 0.0;
    } else if (share > 0) {
        cash = (price - cost) * share;  // sell_stock
        share = 0.0;
    }
    return cash;
}
}  // namespace

TradingResult paper_long_short(
    const vector<vector<double> >& raw_forecasts, const vector<vector<double> >& prices,
    const vector<vector<double> >& costs, int32_t k, bool sign_check, const vector<string>& stock_names,
    double initial_capital
) {
    int32_t n_stocks = (int32_t) raw_forecasts.size();
    TradingResult result{0.0, 0, 0};
    if (n_stocks == 0 || k <= 0 || 2 * k > n_stocks) {
        return result;
    }
    int32_t period = (int32_t) raw_forecasts[0].size();
    for (int32_t i = 0; i < n_stocks; i++) {
        period = std::min(period, (int32_t) raw_forecasts[i].size());
    }
    result.days = period;
    if (period < 1) {
        return result;
    }

    // the Python backtest's company order: stocks sorted by name
    vector<int32_t> by_name(n_stocks);
    std::iota(by_name.begin(), by_name.end(), 0);
    if ((int32_t) stock_names.size() == n_stocks) {
        std::stable_sort(by_name.begin(), by_name.end(), [&](int32_t a, int32_t b) {
            return stock_names[a] < stock_names[b];
        });
    }

    vector<double> share(n_stocks, 0.0);
    double cash = initial_capital;
    vector<int32_t> held;
    vector<int32_t> order(n_stocks);

    for (int32_t t = 0; t < period - 1; t++) {
        // np.argsort(preds[t], kind="stable") over the name-sorted companies: ascending, ties keep name order
        order = by_name;
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            return raw_forecasts[a][t] < raw_forecasts[b][t];
        });
        // longs = order[::-1][:k], shorts = order[:k]
        vector<int32_t> longs, shorts;
        for (int32_t j = 0; j < k; j++) {
            longs.push_back(order[n_stocks - 1 - j]);
            shorts.push_back(order[j]);
        }
        if (sign_check && !(raw_forecasts[longs[k - 1]][t] > 0 && raw_forecasts[shorts[k - 1]][t] < 0)) {
            continue;  // no trade today: keep yesterday's positions
        }
        for (int32_t c : held) {
            cash += clear_holdings(share[c], prices[c][t], costs[c][t]);
        }
        double capital = cash;
        double q_long = capital / k, q_short = capital / k;
        for (int32_t c : longs) {
            share[c] += q_long / (prices[c][t] + costs[c][t]);  // buy_stock
            cash -= q_long;
        }
        for (int32_t c : shorts) {
            double shares_shorted = q_short / (prices[c][t] + costs[c][t]);  // short_stock
            share[c] -= shares_shorted;
            double sold_price = prices[c][t] - costs[c][t];
            cash += shares_shorted * sold_price;
        }
        held = longs;
        held.insert(held.end(), shorts.begin(), shorts.end());
        result.days_traded++;
    }
    int32_t t = period - 1;
    for (int32_t c : by_name) {
        cash += clear_holdings(share[c], prices[c][t], costs[c][t]);
    }
    result.total_return = (cash - initial_capital) / initial_capital;
    return result;
}

// ------------------------------------------------------------------------------------------------ data ----

namespace {
vector<string> split_csv_line(const string& line) {
    vector<string> fields;
    std::stringstream ss(line);
    string field;
    while (std::getline(ss, field, ',')) {
        if (!field.empty() && field.back() == '\r') {
            field.pop_back();
        }
        fields.push_back(field);
    }
    return fields;
}

string stock_of(const string& filename) {
    string base = filename.substr(filename.find_last_of('/') + 1);
    const string suffix = "_val.csv";
    if (base.size() > suffix.size() && base.compare(base.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return base.substr(0, base.size() - suffix.size());
    }
    size_t dot = base.find_last_of('.');
    return dot == string::npos ? base : base.substr(0, dot);
}
}  // namespace

bool load_trading_data(
    const string& trading_file, const vector<string>& validation_filenames, TradingData& data, string& error
) {
    std::ifstream in(trading_file);
    if (!in.good()) {
        error = "could not open trading file '" + trading_file + "'";
        return false;
    }
    string line;
    if (!std::getline(in, line)) {
        error = "trading file '" + trading_file + "' is empty";
        return false;
    }
    vector<string> header = split_csv_line(line);
    map<string, int32_t> column;
    for (int32_t i = 0; i < (int32_t) header.size(); i++) {
        column[header[i]] = i;
    }
    for (const string& needed : {string("RET_MEDIAN"), string("RET_IQR")}) {
        if (column.count(needed) == 0) {
            error = "trading file '" + trading_file + "' has no " + needed + " column";
            return false;
        }
    }

    data = TradingData();
    vector<int32_t> price_column, cost_column;
    for (const string& filename : validation_filenames) {
        string stock = stock_of(filename);
        if (column.count("PRC_" + stock) == 0 || column.count("TC_" + stock) == 0) {
            error = "trading file '" + trading_file + "' has no PRC_" + stock + " / TC_" + stock + " column (from '"
                    + filename + "')";
            return false;
        }
        data.stocks.push_back(stock);
        price_column.push_back(column["PRC_" + stock]);
        cost_column.push_back(column["TC_" + stock]);
    }
    data.prices.assign(data.stocks.size(), vector<double>());
    data.costs.assign(data.stocks.size(), vector<double>());

    int32_t row = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line == "\r") {
            continue;
        }
        vector<string> fields = split_csv_line(line);
        if (fields.size() != header.size()) {
            error = "trading file '" + trading_file + "' row " + std::to_string(row + 1) + " has "
                    + std::to_string(fields.size()) + " fields, the header has " + std::to_string(header.size());
            return false;
        }
        try {
            if (row == 0) {
                data.ret_median = std::stod(fields[column["RET_MEDIAN"]]);
                data.ret_iqr = std::stod(fields[column["RET_IQR"]]);
            }
            for (int32_t i = 0; i < (int32_t) data.stocks.size(); i++) {
                data.prices[i].push_back(std::stod(fields[price_column[i]]));
                data.costs[i].push_back(std::stod(fields[cost_column[i]]));
            }
        } catch (const std::exception&) {
            error = "trading file '" + trading_file + "' row " + std::to_string(row + 1) + ": not a number";
            return false;
        }
        row++;
    }
    if (row < 2) {
        error = "trading file '" + trading_file + "' has fewer than 2 rows";
        return false;
    }
    if (!(data.ret_iqr > 0.0)) {
        error = "trading file '" + trading_file + "': RET_IQR must be positive";
        return false;
    }
    return true;
}

namespace {
std::unique_ptr<TradingData> validation_trading_data;
int32_t trading_k = 10;
}  // namespace

void set_validation_trading_data(const TradingData& data, int32_t k) {
    validation_trading_data.reset(new TradingData(data));
    trading_k = k;
}

const TradingData* get_validation_trading_data() {
    return validation_trading_data.get();
}

int32_t get_trading_k() {
    return trading_k;
}

namespace {
// configuration errors are printed straight to stderr: at start-up EXAMM's log may be restricted to rank 0, and
// a worker that stops first would otherwise end the job without the message being shown
[[noreturn]] void trading_configuration_error(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fflush(stderr);
    exit(1);
}
}  // namespace

void configure_trading_from_arguments(const vector<string>& arguments) {
    string checkpoint, fitness;
    get_argument(arguments, "--checkpoint_metric", false, checkpoint);
    get_argument(arguments, "--fitness_metric", false, fitness);
    bool needed = checkpoint == "return" || fitness == "return";

    string trading_file;
    if (!get_argument(arguments, "--trading_file", false, trading_file)) {
        if (needed) {
            trading_configuration_error(
                "ERROR: --checkpoint_metric return / --fitness_metric return need --trading_file\n"
            );
        }
        return;
    }
    if (argument_exists(arguments, "--validation_sequence_length")) {
        trading_configuration_error(
            "ERROR: --trading_file cannot be used with --validation_sequence_length (each validation series "
            "must be one whole stock)\n"
        );
    }
    vector<string> validation_filenames;
    if (!get_argument_vector(arguments, "--validation_filenames", false, validation_filenames)) {
        trading_configuration_error("ERROR: --trading_file needs --validation_filenames (one file per stock)\n");
    }
    int32_t k = 10;
    get_argument(arguments, "--trading_k", false, k);

    TradingData data;
    string error;
    if (!load_trading_data(trading_file, validation_filenames, data, error)) {
        trading_configuration_error("ERROR: %s\n", error.c_str());
    }
    if (k <= 0 || 2 * k > (int32_t) data.stocks.size()) {
        trading_configuration_error(
            "ERROR: --trading_k %d needs at least %d validation stocks, there are %d\n", k, 2 * k,
            (int32_t) data.stocks.size()
        );
    }
    set_validation_trading_data(data, k);
    Log::info(
        "validation trading return: paper's long-short %d/%d with sign check, costs on, %d stocks, %d rows from '%s'\n",
        k, k, (int32_t) data.stocks.size(), (int32_t) data.prices[0].size(), trading_file.c_str()
    );
}
