#!/usr/bin/env python3
"""Exercise the ported ONE-NAS rules on synthetic data with known answers.

Run from inside the Financial_toolbox checkout, which is where the modules
being tested live:

    cd <financial-toolbox> && /usr/bin/python3 <this file>

The cases are chosen so the correct answer is known without running the
code: a rule that is handed a perfect signal must make money, the same
rule handed the negated signal must lose it, and a dead band must trade
strictly less than the same rule without one.
"""
import os
import sys

import numpy as np

from Logger import Logger
from portfolio import Portfolio
from stock import Stock

N_NAMES, N_DAYS = 50, 60   # the paper trades top/bottom 10 of ~50


def build(seed, signal="perfect", use_tc=False):
    """A panel whose next-day returns are known, with a chosen signal."""
    rng = np.random.default_rng(seed)
    logger = Logger("ERROR")
    portfolio = Portfolio([f"S{i}" for i in range(N_NAMES)])
    portfolio.set_logger(logger)

    rets = rng.normal(0.0, 0.02, size=(N_DAYS, N_NAMES))
    for i in range(N_NAMES):
        s = Stock(f"S{i}")
        s.set_logger(logger)
        price = 100.0 * np.cumprod(1.0 + rets[:, i])
        s.stock_price = price
        s.transaction_cost = np.full(N_DAYS, 0.02)   # 2c per share
        s.ask_price = None
        s.bid_price = None
        # the prediction for day t is next day's realised return, so a
        # perfect signal; negated, it is exactly wrong
        fwd = np.append(rets[1:, i], 0.0)
        s.return_prediction = fwd if signal == "perfect" else -fwd
        s.testing_period = N_DAYS
        s.set_use_TC(use_tc)
        portfolio.add_company_to_protfolio(s)

    portfolio.set_initial_capital(1000.0)
    portfolio.set_use_TC(use_tc)
    return portfolio


def traded_notional(portfolio):
    """Total absolute notional pushed through rebalance_to, for churn tests."""
    return sum(getattr(c, "_churn", 0.0) for c in portfolio.portfolio_list)


def instrument():
    """Wrap rebalance_to so each name accumulates the notional it traded."""
    original = Stock.rebalance_to

    def wrapped(self, target_notional, time):
        before = self.share
        flow = original(self, target_notional, time)
        moved = abs(self.share - before) * abs(self.stock_price[time])
        self._churn = getattr(self, "_churn", 0.0) + moved
        return flow

    Stock.rebalance_to = wrapped


def main():
    instrument()
    failures = []

    def check(label, condition, detail):
        print(f"  {'PASS' if condition else 'FAIL'}  {label}: {detail}")
        if not condition:
            failures.append(label)

    # every strategy is reached through trade(), which is what sets
    # self.strategy for reset(); calling the methods directly is not how
    # this class is used
    def run(portfolio, rule, **kw):
        kw.setdefault("short_company_number", 10)
        return portfolio.trade(rule, **kw)

    print("\nperfect signal should make money, inverted should lose it")
    for rule in ("overlapping_return", "banded_return",
                 "conditional_long_short_return"):
        good = run(build(1, "perfect"), rule)
        bad = run(build(1, "inverted"), rule)
        check(rule, good > 0 > bad, f"perfect {good:+.1f}%, inverted {bad:+.1f}%")

    print("\ncosts must reduce return, never increase it")
    for rule in ("overlapping_return", "banded_return",
                 "conditional_long_short_return"):
        free = run(build(2, "perfect", use_tc=False), rule)
        costed = run(build(2, "perfect", use_tc=True), rule)
        check(rule, costed < free, f"{free:+.1f}% gross vs {costed:+.1f}% net")

    print("\nthe dead band must trade less than an equivalent rule without one")
    p_band = build(3, "perfect")
    run(p_band, "banded_return", entry_rank=10, exit_rank=35)
    banded_churn = traded_notional(p_band)
    p_tight = build(3, "perfect")
    run(p_tight, "banded_return", entry_rank=10, exit_rank=10)   # no band
    tight_churn = traded_notional(p_tight)
    check("banded churn", banded_churn < tight_churn,
          f"band {banded_churn:,.0f} vs no band {tight_churn:,.0f}")

    print("\nnetting: a longer hold must not multiply the notional traded")
    p1 = build(5, "perfect")
    run(p1, "overlapping_return", hold_days=1)
    churn1 = traded_notional(p1)
    p10 = build(5, "perfect")
    run(p10, "overlapping_return", hold_days=10)
    churn10 = traded_notional(p10)
    check("netting reduces churn", churn10 < churn1,
          f"H=10 {churn10:,.0f} vs H=1 {churn1:,.0f}")

    print("\nthe gate must bind when no prediction is negative")
    p = build(6, "perfect")
    for c in p.portfolio_list:
        c.return_prediction = np.abs(c.return_prediction) + 1.0   # all positive
    r = run(p, "conditional_long_short_return")
    check("gate blocks", abs(r) < 1e-9,
          f"all-positive predictions -> {r:+.6f}% (should be flat)")

    print(f"\n{'ALL PASSED' if not failures else 'FAILURES: ' + str(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
