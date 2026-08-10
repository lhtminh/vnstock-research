"""Buffer (hysteresis) rebalancing tests.

The buffer is the difference between a losing strategy and a breakeven one on
this data, and its failure mode is silent: get the survivor logic wrong and it
still produces plausible holdings, just with the turnover it was meant to save.
"""

from __future__ import annotations

import pandas as pd

from vnresearch.backtest import rebalance


def _preds(order_by_date: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Build predictions from an explicit ranking per date, best first."""
    rows = []
    for d, order in order_by_date.items():
        for i, t in enumerate(order):
            rows.append({"date": pd.Timestamp(d), "ticker": t, "pred": float(len(order) - i)})
    dates = pd.DatetimeIndex([pd.Timestamp(d) for d in order_by_date])
    return pd.DataFrame(rows), dates


def test_no_buffer_reproduces_plain_top_n():
    preds, dates = _preds(
        {
            "2024-01-02": ["A", "B", "C", "D", "E"],
            "2024-01-09": ["C", "D", "E", "A", "B"],
        }
    )
    got = rebalance.select(preds, dates, entry_rank=2)
    assert got[dates[0]] == ["A", "B"]
    # Without a buffer the book flips entirely when the ranking flips.
    assert set(got[dates[1]]) == {"C", "D"}


def test_buffer_keeps_a_name_that_only_drifted():
    """B slips from rank 2 to rank 4 — inside the buffer, so it is not sold."""
    preds, dates = _preds(
        {
            "2024-01-02": ["A", "B", "C", "D", "E"],
            "2024-01-09": ["A", "C", "D", "B", "E"],
        }
    )
    got = rebalance.select(preds, dates, entry_rank=2, exit_rank=4)
    assert got[dates[0]] == ["A", "B"]
    assert set(got[dates[1]]) == {"A", "B"}


def test_buffer_sells_once_a_name_falls_past_exit_rank():
    """B drops to rank 5, outside the buffer, so it goes."""
    preds, dates = _preds(
        {
            "2024-01-02": ["A", "B", "C", "D", "E"],
            "2024-01-09": ["A", "C", "D", "E", "B"],
        }
    )
    got = rebalance.select(preds, dates, entry_rank=2, exit_rank=4)
    assert "B" not in got[dates[1]]
    assert set(got[dates[1]]) == {"A", "C"}


def test_book_size_never_exceeds_entry_rank():
    preds, dates = _preds(
        {
            "2024-01-02": ["A", "B", "C", "D", "E", "F"],
            "2024-01-09": ["C", "D", "A", "B", "E", "F"],
            "2024-01-16": ["E", "F", "A", "B", "C", "D"],
        }
    )
    got = rebalance.select(preds, dates, entry_rank=3, exit_rank=6)
    for d in dates:
        assert len(got[d]) <= 3
        assert len(set(got[d])) == len(got[d])


def test_buffer_lowers_turnover():
    """The whole point: same signal, fewer trades.

    Scores drift slowly, which is how a real signal behaves — names shuffle
    around the entry threshold without genuinely changing quality. That churn
    is exactly what the buffer is meant to ignore.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    names = [f"T{i}" for i in range(30)]
    score = rng.normal(0, 1, len(names))

    order = {}
    for k in range(20):
        score = score + rng.normal(0, 0.3, len(names))  # slow drift
        ranked = [names[i] for i in np.argsort(-score)]
        order[str(pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * k))[:10]] = ranked
    preds, dates = _preds(order)

    plain = rebalance.turnover(rebalance.select(preds, dates, entry_rank=5))
    buffered = rebalance.turnover(rebalance.select(preds, dates, entry_rank=5, exit_rank=15))
    assert buffered < plain, f"buffered {buffered:.3f} not below plain {plain:.3f}"


def test_exit_rank_below_entry_rank_is_rejected():
    preds, dates = _preds({"2024-01-02": ["A", "B", "C"]})
    try:
        rebalance.select(preds, dates, entry_rank=3, exit_rank=1)
    except ValueError as e:
        assert "exit_rank" in str(e)
    else:
        raise AssertionError("expected ValueError")
