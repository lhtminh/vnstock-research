"""Beta-hedge overlay tests.

The hedge must do two things and no more: remove the market, and not invent
alpha that was never there. A hedge that silently uses future information does
both jobs beautifully and is worthless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vnresearch.backtest import hedge


def _market(n=800, seed=0, mu=0.0005, sd=0.011):
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=n))
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, n), index=idx)


def test_pure_beta_book_hedges_to_roughly_zero():
    """A book that is ONLY market exposure must net out to about nothing.

    This is the whole point: if the strategy has no skill, hedging must reveal
    that rather than hide it.
    """
    m = _market()
    book = 0.8 * m
    h = hedge.apply(book, m)
    ann = (1 + h.hedged).prod() ** (252 / len(h.hedged)) - 1
    assert abs(ann) < 0.02


def test_hedge_recovers_a_known_alpha():
    """Beta 0.8 plus a steady 4bp a day must survive the hedge."""
    m = _market()
    daily_alpha = 0.0004
    book = 0.8 * m + daily_alpha
    h = hedge.apply(book, m)
    ann = (1 + h.hedged).prod() ** (252 / len(h.hedged)) - 1
    expected = (1 + daily_alpha) ** 252 - 1
    assert ann == pytest.approx(expected, abs=0.02)


def test_hedge_removes_market_exposure():
    m = _market()
    book = 1.2 * m + 0.0002
    h = hedge.apply(book, m)
    resid = float(
        np.cov(h.hedged, m.reindex(h.hedged.index))[0, 1] / np.var(m.reindex(h.hedged.index))
    )
    assert abs(resid) < 0.15


def test_hedge_lowers_volatility():
    m = _market()
    book = 0.9 * m + pd.Series(np.random.default_rng(1).normal(0, 0.004, len(m)), index=m.index)
    h = hedge.apply(book, m)
    assert h.hedged.std() < h.gross.std()


def test_hedge_ratio_is_point_in_time():
    """The ratio applied on day t must not depend on day t.

    Feeding the hedge a market that only starts moving late must leave the
    early hedge ratio untouched — if a full-sample beta leaked in, the early
    ratio would already reflect the later regime.
    """
    m = _market()
    book = 0.8 * m
    early = hedge.apply(book.iloc[:400], m.iloc[:400])
    full = hedge.apply(book, m)
    pd.testing.assert_series_equal(early.beta, full.beta.iloc[:400], check_names=False, rtol=1e-9)


def test_costs_are_charged_and_reduce_return():
    m = _market()
    book = 0.8 * m + 0.0004
    free = hedge.apply(book, m, cost_per_side=0.0, rolls_per_year=0)
    paid = hedge.apply(book, m, cost_per_side=0.001, rolls_per_year=12)
    assert paid.cost.sum() > 0
    assert paid.hedged.sum() < free.hedged.sum()


def test_hedge_ratio_is_clipped_to_sane_bounds():
    """A noisy window can produce an absurd beta; never short 5x the book."""
    m = _market(sd=0.001)
    book = pd.Series(
        np.random.default_rng(2).normal(0, 0.05, len(m)), index=m.index
    )  # book unrelated to market
    h = hedge.apply(book, m)
    assert h.beta.max() <= 2.0
    assert h.beta.min() >= 0.0
