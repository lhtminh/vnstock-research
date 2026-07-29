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

    w = engine._target_weights(preds, dates, tickers, 2, 5, adtv, 1e9, 0.10)
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

    w = engine._target_weights(preds, dates, tickers, 2, 5, adtv, 1e9, 0.10)
    assert w.iloc[0]["A"] == pytest.approx(0.01)
    assert w.iloc[0]["B"] == pytest.approx(0.5)


def test_weights_hold_between_rebalances():
    dates = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=10))
    tickers = ["A", "B"]
    preds = _preds(dates, tickers, {"A": 2, "B": 1})
    adtv = pd.DataFrame(1e12, index=dates, columns=tickers)

    w = engine._target_weights(preds, dates, tickers, 1, 5, adtv, 1e9, 0.10)
    # Rebalance on day 0 and day 5; days 1-4 must carry the position, not sell it.
    assert (w.iloc[0:5]["A"] > 0).all()
    assert not np.isnan(w.to_numpy()).any()
