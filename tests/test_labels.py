"""Label alignment tests on a synthetic panel — no database needed.

The failure these guard against is silent: a label that is off by one session,
or that reads backwards, still trains and still backtests. It just produces a
strategy that cannot be traded.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from vnresearch.label import forward


def _panel(tmp_path, rows) -> str:
    df = pd.DataFrame(rows)
    path = tmp_path / "panel.parquet"
    df.to_parquet(path)
    return path.as_posix()


def _run(panel_path: str, horizon: int = 2, price_col: str = "open"):
    sql = forward._sql([horizon], horizon, price_col).replace("{panel}", panel_path)
    con = duckdb.connect()
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def _straight_line(n=10, start="2026-01-05"):
    """One ticker, consecutive business days, open rising 100 a day."""
    dates = pd.bdate_range(start, periods=n).date
    return [
        {
            "ticker": "AAA",
            "date": d,
            "open": 10_000.0 + 100 * i,
            "close": 10_000.0 + 100 * i,
            "in_universe": True,
            "tradeable": True,
            "bar_status": "normal",
            "adtv": 5e9,
        }
        for i, d in enumerate(dates)
    ]


def test_entry_is_the_next_session_not_today(tmp_path):
    df = _run(_panel(tmp_path, _straight_line()))
    first = df.iloc[0]
    # Signal on day 0 must buy at day 1's open, never day 0's.
    assert first["entry_px"] == 10_100.0
    assert first["entry_date"] > first["date"]


def test_forward_return_spans_exactly_the_horizon(tmp_path):
    df = _run(_panel(tmp_path, _straight_line()), horizon=2)
    first = df.iloc[0]
    # entry at index 1 (10,100), exit at index 1+2 = 3 (10,300).
    # Exit columns carry the horizon suffix; only the primary is aliased.
    assert first["exit_px_2"] == 10_300.0
    assert first["fwd_ret"] == pytest.approx(10_300 / 10_100 - 1)


def test_label_looks_forward_not_backward(tmp_path):
    """On a rising series every label must be positive."""
    df = _run(_panel(tmp_path, _straight_line()))
    valid = df[df["label_ok"]]
    assert len(valid) > 0
    assert (valid["fwd_ret"] > 0).all()


def test_untradeable_entry_invalidates_the_label(tmp_path):
    rows = _straight_line()
    rows[1]["tradeable"] = False  # cannot buy at the entry bar
    rows[1]["bar_status"] = "limit_up"
    df = _run(_panel(tmp_path, rows))
    assert not df.iloc[0]["label_ok"]
    assert pd.isna(df.iloc[0]["fwd_ret"])


def test_untradeable_exit_still_labels(tmp_path):
    """Deliberate: dropping limit-up exits would truncate the winning tail."""
    rows = _straight_line()
    rows[3]["tradeable"] = False
    rows[3]["bar_status"] = "limit_up"
    df = _run(_panel(tmp_path, rows), horizon=2)
    row = df.iloc[0]
    assert row["label_ok"]
    assert not row["exit_tradeable_2"]


def test_long_suspension_gap_is_not_labelled(tmp_path):
    """A ticker that stops trading for months must not price an 'exit' later."""
    rows = _straight_line(n=4)
    rows[1]["date"] = pd.Timestamp("2026-06-01").date()  # months after row 0
    rows[2]["date"] = pd.Timestamp("2026-06-02").date()
    rows[3]["date"] = pd.Timestamp("2026-06-03").date()
    df = _run(_panel(tmp_path, rows), horizon=2)
    assert not df.iloc[0]["label_ok"]


def test_rank_is_within_the_day_only(tmp_path):
    """Two tickers, opposite outcomes: the winner ranks 1.0, the loser 0.0."""
    dates = pd.bdate_range("2026-01-05", periods=6).date
    rows = []
    for i, d in enumerate(dates):
        rows.append(
            {
                "ticker": "UP",
                "date": d,
                "open": 10_000.0 + 500 * i,
                "close": 10_000.0 + 500 * i,
                "in_universe": True,
                "tradeable": True,
                "bar_status": "normal",
                "adtv": 5e9,
            }
        )
        rows.append(
            {
                "ticker": "DN",
                "date": d,
                "open": 10_000.0 - 300 * i,
                "close": 10_000.0 - 300 * i,
                "in_universe": True,
                "tradeable": True,
                "bar_status": "normal",
                "adtv": 5e9,
            }
        )
    df = _run(_panel(tmp_path, rows), horizon=2)
    # DuckDB returns dates as datetime64, so normalise before comparing.
    day0 = df[(df["date"] == pd.Timestamp(dates[0])) & df["label_ok"]]
    assert day0.set_index("ticker").loc["UP", "fwd_rank"] == 1.0
    assert day0.set_index("ticker").loc["DN", "fwd_rank"] == 0.0
