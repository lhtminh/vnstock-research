"""Cost model and weighting tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vnresearch.backtest import engine
from vnresearch.backtest.costs import Costs


def test_round_trip_cost_is_above_half_a_percent():
    """The number that decides which signals are tradeable in Vietnam."""
    c = Costs(brokerage=0.0015, sell_tax=0.0010, slippage=0.0010)
    assert c.round_trip == pytest.approx(0.006)
    assert c.round_trip > 0.005


def test_sell_tax_is_split_across_both_sides():
    c = Costs(brokerage=0.0015, sell_tax=0.0010, slippage=0.0)
    # Charging half each way costs the same over a round trip as the real
    # sales-only tax, and keeps vectorbt to one symmetric fee.
    assert 2 * c.symmetric_fee == pytest.approx(2 * c.brokerage + c.sell_tax)


def test_breakeven_scales_with_turnover():
    c = Costs.load()
    assert c.breakeven_spread(1.0) > c.breakeven_spread(0.5)
    assert c.breakeven_spread(0.0) == 0.0


def _preds(dates, tickers, scores):
    return pd.DataFrame(
        [{"date": d, "ticker": t, "pred": scores[t]} for d in dates for t in tickers]
    )


def test_weights_pick_the_top_n_and_sum_to_one():
    dates = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=10))
    tickers = ["A", "B", "C", "D", "E"]
    preds = _preds(dates, tickers, {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1})
    adtv = pd.DataFrame(1e12, index=dates, columns=tickers)  # capacity not binding
    equity = pd.Series(1e9, index=dates)

    w = engine._target_weights(preds, dates, tickers, 2, 5, adtv, equity, 0.10)
    row = w.iloc[0]
    assert row["A"] == pytest.approx(0.5)
    assert row["B"] == pytest.approx(0.5)
    assert row[["C", "D", "E"]].sum() == 0.0
    assert row.sum() == pytest.approx(1.0)


def test_capacity_cap_shrinks_positions_in_thin_names():
    dates = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=5))
    tickers = ["A", "B"]
    preds = _preds(dates, tickers, {"A": 2, "B": 1})
    # A trades 100m VND/day; 10% of that is 10m against 1bn capital = 1% weight,
    # far below the 50% an equal split would give it.
    adtv = pd.DataFrame({"A": 1e8, "B": 1e12}, index=dates)
    equity = pd.Series(1e9, index=dates)

    w = engine._target_weights(preds, dates, tickers, 2, 5, adtv, equity, 0.10)
    assert w.iloc[0]["A"] == pytest.approx(0.01)
    assert w.iloc[0]["B"] == pytest.approx(0.5)


def test_capacity_tightens_as_the_book_grows():
    """The cap is a VND limit, so it must scale with equity, not starting cash.

    This was a real bug: using init_cash meant a book that compounded 7x kept
    a cap sized for its first day, silently allowing 70% of ADTV.
    """
    dates = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=5))
    tickers = ["A", "B"]
    preds = _preds(dates, tickers, {"A": 2, "B": 1})
    adtv = pd.DataFrame({"A": 1e8, "B": 1e12}, index=dates)

    small = engine._target_weights(
        preds, dates, tickers, 2, 5, adtv, pd.Series(1e9, index=dates), 0.10
    )
    big = engine._target_weights(
        preds, dates, tickers, 2, 5, adtv, pd.Series(7e9, index=dates), 0.10
    )
    assert big.iloc[0]["A"] < small.iloc[0]["A"]
    assert big.iloc[0]["A"] == pytest.approx(0.10 * 1e8 / 7e9)


def test_index_tracker_shows_beta_one_and_no_alpha():
    """Holding the index must report beta 1 and alpha 0, not a 'result'."""
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=600))
    rng = np.random.default_rng(0)
    br = rng.normal(0.0004, 0.01, len(idx))
    bench = pd.Series(1000 * np.cumprod(1 + br), index=idx)

    p = engine.performance(pd.Series(1e9 * np.cumprod(1 + br), index=idx), bench)
    assert p["beta"] == pytest.approx(1.0, abs=0.01)
    assert p["alpha"] == pytest.approx(0.0, abs=0.005)
    assert p["correlation"] == pytest.approx(1.0, abs=0.01)


def test_leverage_is_beta_not_alpha():
    """2x the index is beta 2 and NEGATIVE alpha, from volatility drag.

    Compounding 2r loses sigma^2 a year against twice the compounded r. Doubling
    exposure buys you none of the drag back, which is exactly why leverage must
    never be allowed to show up as skill.
    """
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=600))
    rng = np.random.default_rng(0)
    br = rng.normal(0.0004, 0.01, len(idx))
    bench = pd.Series(1000 * np.cumprod(1 + br), index=idx)

    p = engine.performance(pd.Series(1e9 * np.cumprod(1 + 2 * br), index=idx), bench)
    assert p["beta"] == pytest.approx(2.0, abs=0.02)
    annual_var = float(np.var(br) * 252)
    assert p["alpha"] == pytest.approx(-annual_var, abs=0.005)


def test_annual_table_reports_excess_per_year():
    idx = pd.DatetimeIndex(pd.bdate_range("2021-01-01", periods=520))
    bench = pd.Series(np.linspace(1000, 1100, len(idx)), index=idx)
    equity = pd.Series(np.linspace(1e9, 1.3e9, len(idx)), index=idx)

    t = engine.annual_table(equity, bench)
    assert list(t.columns) == ["strategy", "benchmark", "excess"]
    assert len(t) == 2  # two calendar years
    assert (t["excess"] == t["strategy"] - t["benchmark"]).all()


def test_weights_hold_between_rebalances():
    dates = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=10))
    tickers = ["A", "B"]
    preds = _preds(dates, tickers, {"A": 2, "B": 1})
    adtv = pd.DataFrame(1e12, index=dates, columns=tickers)

    w = engine._target_weights(preds, dates, tickers, 1, 5, adtv, pd.Series(1e9, index=dates), 0.10)
    # Rebalance on day 0 and day 5; days 1-4 must carry the position, not sell it.
    assert (w.iloc[0:5]["A"] > 0).all()
    assert not np.isnan(w.to_numpy()).any()


def test_book_never_exceeds_one_hundred_percent():
    """Positions must be replaced at each rebalance, not accumulated.

    This was a real bug: blanking zeros before ffill made dropped names carry
    their old weight forever, so every rebalance added N more holdings. The book
    reached 1,156% deployed before it was caught.
    """
    dates = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=60))
    tickers = [f"T{i}" for i in range(20)]
    adtv = pd.DataFrame(1e12, index=dates, columns=tickers)
    equity = pd.Series(1e9, index=dates)

    # A different winner every rebalance, so nothing is ever re-picked.
    rows = []
    for i, d in enumerate(dates):
        for j, t in enumerate(tickers):
            rows.append({"date": d, "ticker": t, "pred": 1.0 if j == (i // 5) % 20 else 0.0})
    preds = pd.DataFrame(rows)

    # top_n=1 so the pick is unambiguous; with ties the runner-up is arbitrary.
    w = engine._target_weights(preds, dates, tickers, 1, 5, adtv, equity, 0.10)
    assert w.sum(axis=1).max() == pytest.approx(1.0)
    # And the name picked first must actually be gone once it stops winning.
    assert w.iloc[-1]["T0"] == 0.0
