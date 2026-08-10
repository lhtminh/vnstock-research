"""The scoring metrics, and specifically why there are two of them.

`daily_ic` scores the ordering of the whole cross-section — ~280 names on a
recent day. The book buys 50. So a model can rank the middle better and the top
worse and still post a higher IC, which is exactly what happened going from 35
features to 59: walk-forward IC rose +0.0730 -> +0.0825 while dev alpha fell
-2.93% -> -5.18%.

`topn_edge` scores the 50 that get bought. These tests pin the difference so
neither metric can quietly start measuring the other one's question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vnresearch.model.dataset import daily_ic, topn_edge

N_DAYS = 40
N_NAMES = 300


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.Series(np.repeat(pd.bdate_range("2024-01-01", periods=N_DAYS), N_NAMES))
    fwd = rng.normal(0, 0.05, N_DAYS * N_NAMES)
    return rng, dates, fwd


def test_a_perfect_ranker_earns_a_large_edge():
    _, dates, fwd = _frame()
    assert topn_edge(fwd, fwd, dates, n=50).mean() > 0.05


def test_a_random_ranker_earns_nothing():
    rng, dates, fwd = _frame()
    noise = rng.normal(size=N_DAYS * N_NAMES)
    assert abs(topn_edge(noise, fwd, dates, n=50).mean()) < 0.005


def test_an_inverted_ranker_loses_symmetrically():
    _, dates, fwd = _frame()
    good = topn_edge(fwd, fwd, dates, n=50).mean()
    bad = topn_edge(-fwd, fwd, dates, n=50).mean()
    assert bad < -0.05
    assert abs(good + bad) < 0.005, "picking the worst should mirror picking the best"


def test_the_edge_is_measured_against_the_day_not_the_sample():
    """Subtracting the day's own universe mean removes the market move for that
    window exactly. A day where everything rose must not read as skill."""
    dates = pd.Series(np.repeat(pd.bdate_range("2024-01-01", periods=4), N_NAMES))
    rng = np.random.default_rng(1)
    fwd = rng.normal(0, 0.02, 4 * N_NAMES)
    flat = topn_edge(rng.normal(size=len(fwd)), fwd, dates, n=50).mean()
    # Same returns, but every day shifted up 10%. A metric measured against
    # zero would report a huge edge; measured against the day, nothing changes.
    shifted = topn_edge(rng.normal(size=len(fwd)), fwd + 0.10, dates, n=50).mean()
    assert abs(flat - shifted) < 0.01


def test_ic_and_top_edge_can_disagree():
    """The reason both metrics exist, made concrete.

    A ranker that orders the bulk of the cross-section well but promotes the
    genuinely worst names to the very top posts a strong IC and a NEGATIVE
    traded edge. On IC alone this model looks good; it would lose money on
    every rebalance.

    A wide universe is the point: with 600 names, misplacing 50 of them barely
    dents a rank correlation computed over all 600, while completely determining
    what the book holds. That asymmetry is the real one — the live universe is
    ~280 names and the book takes 50 of them.
    """
    n_names = 600
    dates = pd.Series(np.repeat(pd.bdate_range("2024-01-01", periods=N_DAYS), n_names))
    rng = np.random.default_rng(2)
    fwd = rng.normal(0, 0.05, N_DAYS * n_names)

    df = pd.DataFrame({"d": dates.to_numpy(), "r": fwd})
    df["p"] = df["r"]
    # Give the 50 WORST names of each day the highest predictions.
    for _, g in df.groupby("d"):
        worst = g.nsmallest(50, "r").index
        df.loc[worst, "p"] = g["r"].max() + 1.0

    ic = daily_ic(df["p"].to_numpy(), df["r"].to_numpy(), df["d"]).mean()
    edge = topn_edge(df["p"].to_numpy(), df["r"].to_numpy(), df["d"], n=50).mean()
    assert ic > 0.5, f"the bulk ordering is still strong (got IC {ic:.3f})"
    assert edge < -0.05, f"yet the book buys the worst names (got edge {edge:+.4f})"


def test_a_universe_no_wider_than_the_book_is_not_scored():
    """With 50 names and a top-50 book, 'the top 50' is everything and the edge
    is 0 by construction. Returning NaN says 'not measurable' instead."""
    dates = pd.Series(np.repeat(pd.bdate_range("2024-01-01", periods=5), 60))
    rng = np.random.default_rng(3)
    fwd = rng.normal(0, 0.05, 5 * 60)
    assert topn_edge(fwd, fwd, dates, n=50).isna().all()
