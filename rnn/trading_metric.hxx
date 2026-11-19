#ifndef EXAMM_TRADING_METRIC_HXX
#define EXAMM_TRADING_METRIC_HXX

#include <cstdint>
#include <string>
using std::string;

#include <vector>
using std::vector;

/**
 * The trading-return metric: the validation net return of the paper's Algorithm 1 (daily long-short), usable as
 * the checkpoint metric and/or the fitness (EvaluationMetric::RETURN). It reproduces Financial_toolbox's
 * paper_long_short.py with stock.py's transaction costs (costs on both legs), so a genome's in-search return is the
 * number the Python backtest reports for the same forecasts.
 *
 * Algorithm 1, for days t = 0 .. n-2 (forecast t is made at the close of day t for the return t -> t+1):
 *   rank the stocks by forecast; longs = the k highest, shorts = the k lowest;
 *   if sign_check and not (k-th highest forecast > 0 and k-th lowest forecast < 0): keep yesterday's positions;
 *   otherwise close every held position, then put cash / k into each long and short.
 * Day n-1: close everything. Trades are priced at price +/- cost per share (buys pay price + cost, sells and short
 * sales receive price - cost; a long buys cash / (price + cost) shares, a short sells the same count).
 * Ties in the ranking are broken by stock name, as the Python backtest's stable sort over sorted stock names.
 */

struct TradingResult {
    double total_return;  // (final cash - initial capital) / initial capital, e.g. 0.05 = +5 %
    int32_t days_traded;  // days the book was (re)formed
    int32_t days;         // forecast days used (n)
};

// raw_forecasts[stock][day] (raw daily-return forecasts; only their signs and order matter), prices[stock][day],
// costs[stock][day] (per share). Every stock needs at least n = raw_forecasts[stock].size() prices and costs.
// stock_names gives the tie-break order (empty: index order).
TradingResult paper_long_short(
    const vector<vector<double> >& raw_forecasts, const vector<vector<double> >& prices,
    const vector<vector<double> >& costs, int32_t k, bool sign_check, const vector<string>& stock_names,
    double initial_capital = 1000.0
);

// Prices, costs and forecast scaling for the validation series, in validation-series order.
struct TradingData {
    vector<string> stocks;           // stock of each validation series
    vector<vector<double> > prices;  // [series][row of the validation file]
    vector<vector<double> > costs;
    double ret_median = 0.0;  // raw return forecast = RET forecast * ret_iqr + ret_median
    double ret_iqr = 1.0;
};

// Reads a trading_validation.csv (scripts/data_prep/write_core7_trading_files.py) and orders it like the validation
// series: series i is the stock named by validation_filenames[i] ("<path>/<STOCK>_val.csv"). Returns false and
// sets error if the file or a stock is missing.
bool load_trading_data(
    const string& trading_file, const vector<string>& validation_filenames, TradingData& data, string& error
);

// Process-wide validation trading data (the same in every MPI rank / thread; loaded once at start-up).
void set_validation_trading_data(const TradingData& data, int32_t k);
const TradingData* get_validation_trading_data();  // nullptr if not configured
int32_t get_trading_k();

// Called by examm_mpi / examm_mt on every rank after the data are loaded:
//   --trading_file <trading_validation.csv> [--trading_k 10]
// Exits with an error if --fitness_metric return or --checkpoint_metric return is given without --trading_file,
// or if the file does not match the validation files. Does nothing if neither is used.
void configure_trading_from_arguments(const vector<string>& arguments);

#endif
