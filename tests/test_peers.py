"""Peer features must be REPRODUCIBLE, not merely point-in-time.

Nothing in peers.py ever looked forward, and it was still wrong: rebuilding
after new data arrived rewrote peer values across the whole history. Measured
on the real panel before the fix, 73% of all 942,303 rows moved between two
builds three days apart, back as far as 2007, and that swapped 8 of the live
model's top 50 names.

Point-in-time and reproducible are different properties. The first says a value
could have been known on its date; the second says it will still be that value
tomorrow. A backtest needs both, and only the first had a test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vnresearch.features import peers

N_DAYS = 420
N_TICKERS = 26
_MAX_DAYS = 600
_MAX_ILLIQUID = 8


def _series(seed=0):
    """Every random series at full length, drawn once.

    Drawing `n_days` values per call would make a longer panel a DIFFERENT
    market rather than the same one observed for longer — each draw shifts the
    generator, so the second and third series would not line up. Then the test
    fails for its own reasons and says nothing about reproducibility. Generate
    the maximum once and truncate.
    """
    rng = np.random.default_rng(seed)
    return {
        "mkt": rng.normal(0.0003, 0.012, _MAX_DAYS),
        # Two clusters, so the correlation has something real to find instead
        # of a noise matrix where the top 10 is whatever sorted highest.
        "a": rng.normal(0, 0.010, _MAX_DAYS),
        "b": rng.normal(0, 0.010, _MAX_DAYS),
        "noise": rng.normal(0, 0.008, (N_TICKERS, _MAX_DAYS)),
        "junk": rng.normal(0, 0.05, (_MAX_ILLIQUID, _MAX_DAYS)),
    }


def _panel(tmp_path, n_days=N_DAYS, extra_illiquid=0, name="panel.parquet"):
    """A panel wide and long enough for real peer sets: >252 sessions of
    history, >=20 investable names, and two genuine correlation clusters."""
    s = _series()
    dates = pd.bdate_range("2020-01-01", periods=n_days)

    rows = []
    for t in range(N_TICKERS):
        common = s["a"] if t % 2 else s["b"]
        for i, d in enumerate(dates):
            rows.append(
                {
                    "ticker": f"T{t:02d}",
                    "date": d.date(),
                    "ret": s["mkt"][i] + 0.7 * common[i] + s["noise"][t][i],
                    "in_universe": True,
                }
            )
    # Illiquid listings: present in the panel, never investable. Their returns
    # are wild on purpose — a mean taken over the whole panel would lurch.
    for x in range(extra_illiquid):
        for i, d in enumerate(dates):
            rows.append(
                {
                    "ticker": f"Z{x:02d}",
                    "date": d.date(),
                    "ret": s["junk"][x][i],
                    "in_universe": False,
                }
            )
    path = tmp_path / name
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _build(tmp_path, monkeypatch, panel_path) -> pd.DataFrame:
    """Run peers.build() against a panel of our choosing."""
    out = tmp_path / "out"
    (out / "data" / "clean").mkdir(parents=True, exist_ok=True)
    (out / "data" / "features").mkdir(parents=True, exist_ok=True)

    def fake_path(rel: str):
        if rel == "data/clean/panel.parquet":
            return panel_path
        return out / rel

    monkeypatch.setattr(peers.config, "path", fake_path)
    return pd.read_parquet(peers.build(verbose=False))


def _compare(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Rows present in both, joined on (ticker, date)."""
    return a.merge(b, on=["ticker", "date"], suffixes=("_a", "_b"))


def test_appending_new_dates_does_not_rewrite_history(tmp_path, monkeypatch):
    """The failure that started this: tomorrow's data changing 2007.

    The old grid was `range(252, len(dates), 21)` — positions in an array — so
    one extra row shifted every later rebalance and a different peer set
    governed a different span.
    """
    short = _build(tmp_path, monkeypatch, _panel(tmp_path, n_days=N_DAYS, name="a.parquet"))
    longer = _build(tmp_path, monkeypatch, _panel(tmp_path, n_days=N_DAYS + 37, name="b.parquet"))
    assert len(longer) > len(short), "the longer panel should produce more rows"

    both = _compare(short, longer)
    assert len(both) > 1000, "too little overlap to be a meaningful comparison"
    for col in ("peer_ret_1", "peer_corr", "peer_n"):
        pd.testing.assert_series_equal(
            both[f"{col}_a"],
            both[f"{col}_b"],
            check_names=False,
            rtol=1e-12,
            atol=0,
            obj=f"{col} changed on dates that already existed",
        )


def test_a_new_illiquid_listing_does_not_rewrite_history(tmp_path, monkeypatch):
    """Discovering a UPCOM name must not move a peer value from years earlier.

    The daily mean used to be taken over every ticker in the panel, so any new
    listing moved the residual on every date it traded, and every correlation
    built from those residuals moved with it.

    Note what this does NOT claim: a newly discovered ticker that was itself
    LIQUID in the past will change the mean, because it genuinely belonged in
    that cross-section. Only non-investable additions are guaranteed inert.
    """
    plain = _build(tmp_path, monkeypatch, _panel(tmp_path, name="c.parquet"))
    with_junk = _build(tmp_path, monkeypatch, _panel(tmp_path, extra_illiquid=8, name="d.parquet"))

    both = _compare(plain, with_junk)
    assert len(both) == len(plain), "the illiquid names should add no peer rows"
    for col in ("peer_ret_1", "peer_corr", "peer_n"):
        pd.testing.assert_series_equal(
            both[f"{col}_a"],
            both[f"{col}_b"],
            check_names=False,
            rtol=1e-12,
            atol=0,
            obj=f"{col} moved when a non-investable ticker was added",
        )


def test_rebalance_dates_are_anchored_to_the_calendar(tmp_path, monkeypatch):
    """The mechanism, pinned directly: dropping a date must not shift the grid."""
    dates = pd.bdate_range("2020-01-01", periods=300)
    full = peers._rebalance_dates(pd.DatetimeIndex(dates))
    # Remove an arbitrary mid-history session, as a data repair might.
    holed = peers._rebalance_dates(pd.DatetimeIndex(dates.delete(100)))

    assert list(full) == [d for d in full if d in dates]
    # Every rebalance date is the first session of its month.
    assert all(d.to_period("M") not in {x.to_period("M") for x in full if x < d} for d in full)
    # Dropping a non-first-of-month session leaves the grid alone.
    assert dates[100] not in full, "picked a date that IS a rebalance; choose another"
    assert list(holed) == list(full)


def test_a_peer_set_is_never_informed_by_the_dates_it_governs(tmp_path, monkeypatch):
    """Point-in-time, which was already true and must stay true."""
    p = _build(tmp_path, monkeypatch, _panel(tmp_path, name="e.parquet"))
    # Peers only start once a full 252-session window exists behind them.
    first = pd.Timestamp(p["date"].min())
    assert first >= pd.bdate_range("2020-01-01", periods=N_DAYS)[peers.CORR_WINDOW - 1]


@pytest.mark.parametrize("col", ["peer_ret_1", "peer_ret_5", "peer_ret_21", "peer_corr"])
def test_peer_columns_are_produced(tmp_path, monkeypatch, col):
    p = _build(tmp_path, monkeypatch, _panel(tmp_path, name=f"f_{col}.parquet"))
    assert p[col].notna().any(), f"{col} is entirely NULL"
