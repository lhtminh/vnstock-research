"""Feature correctness tests.

The headline test is truncation invariance: features computed on history up to
day T must be identical whether or not the data after T exists. Any feature
that peeks forward fails it. This catches leakage that eyeballing the SQL
does not, because a leak looks like an ordinary window until you compare.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from vnresearch.features import build
from vnresearch.features.registry import all_features


def _synthetic_panel(tmp_path, n_days=400, n_tickers=6, seed=0) -> str:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days).date
    mkt = rng.normal(0.0004, 0.011, n_days)
    rows = []
    for t in range(n_tickers):
        px = 20_000.0
        prev = None
        for i, d in enumerate(dates):
            r = 0.8 * mkt[i] + rng.normal(0, 0.015)
            px = max(px * (1 + r), 1_000.0)
            vol = int(rng.integers(50_000, 500_000))
            rows.append(
                {
                    "ticker": f"T{t}",
                    "date": d,
                    "open": px * (1 + rng.normal(0, 0.002)),
                    "high": px * 1.01,
                    "low": px * 0.99,
                    "close": px,
                    "volume": vol,
                    "exchange": "HOSE",
                    "symbol_status": "active",
                    "bar_status": "normal",
                    "prev_close": prev,
                    "ret": None if prev is None else px / prev - 1,
                    "turnover": px * vol,
                    "adtv": px * vol,
                    "adtv_obs": 20,
                    "tradeable": True,
                    "in_universe": True,
                    "mkt_ret": mkt[i],
                    # Trailing-year share issuance, joined in by the panel.
                    # Non-zero so the feature is exercised rather than skipped.
                    "dilution_252": 0.01 * (i % 5),
                    # Seasonality and peer columns, also joined in upstream.
                    # Varied per ticker and date so the features actually
                    # produce values instead of a constant the tests cannot see.
                    "seas_month": 0.0002 * ((t + i) % 7 - 3),
                    "seas_tet": 0.0003 * ((t + i) % 5 - 2),
                    "days_from_tet": (i % 250) - 20,
                    "peer_ret_1": 0.7 * mkt[i],
                    "peer_ret_5": 0.7 * mkt[max(i - 4, 0) : i + 1].sum(),
                    "peer_ret_21": 0.7 * mkt[max(i - 20, 0) : i + 1].sum(),
                    "peer_corr": 0.3 + 0.01 * (t % 5),
                    "peer_n": 10,
                }
            )
            prev = px
    path = tmp_path / "panel.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path.as_posix()


def _compute(panel_path: str) -> pd.DataFrame:
    sql, _ = build._raw_sql(panel_path, join_peers=False)
    con = duckdb.connect()
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def test_no_feature_looks_forward_in_its_sql():
    """A window frame may never reference FOLLOWING, and LEAD is banned outright."""
    for name, f in all_features().items():
        upper = f.sql.upper()
        assert "FOLLOWING" not in upper, f"{name} has a forward-looking frame"
        assert "LEAD(" not in upper, f"{name} uses LEAD"


def test_features_are_truncation_invariant(tmp_path):
    """Cut the data in half; every feature value before the cut must be unchanged."""
    full_path = _synthetic_panel(tmp_path)
    full = _compute(full_path)

    cut = pd.Timestamp("2024-01-02")
    truncated_df = pd.read_parquet(full_path)
    truncated_df = truncated_df[pd.to_datetime(truncated_df["date"]) < cut]
    trunc_path = tmp_path / "panel_trunc.parquet"
    truncated_df.to_parquet(trunc_path)
    trunc = _compute(trunc_path.as_posix())

    names = sorted(all_features().keys())
    a = full[full["date"] < cut].sort_values(["ticker", "date"]).reset_index(drop=True)
    b = trunc.sort_values(["ticker", "date"]).reset_index(drop=True)
    assert len(a) == len(b) and len(a) > 0

    for n in names:
        pd.testing.assert_series_equal(
            a[n],
            b[n],
            check_names=False,
            rtol=1e-9,
            atol=1e-12,
            obj=f"{n} changed when future data was removed",
        )


def test_every_feature_produces_some_values(tmp_path):
    """An all-NULL feature is a broken expression, not a data boundary."""
    df = _compute(_synthetic_panel(tmp_path))
    for n in sorted(all_features().keys()):
        assert df[n].notna().any(), f"{n} is entirely NULL on clean synthetic data"


def test_warmup_is_blanked_not_guessed(tmp_path):
    """A 252-session feature must be NULL on a ticker's first bars."""
    df = _compute(_synthetic_panel(tmp_path))
    first = df.sort_values(["ticker", "date"]).groupby("ticker").head(10)
    assert first["mom_12_1"].isna().all()
    assert first["ret_252"].isna().all()


def test_lookback_matches_first_non_null(tmp_path):
    """The declared lookback should be where values actually start appearing."""
    df = _compute(_synthetic_panel(tmp_path)).sort_values(["ticker", "date"])
    one = df[df["ticker"] == "T0"].reset_index(drop=True)
    for name, f in all_features().items():
        vals = one[name]
        if not vals.notna().any():
            continue
        first_idx = int(vals.first_valid_index())
        # first_idx is 0-based, lookback counts sessions, so allow the offset.
        assert first_idx >= f.lookback - 1, f"{name} appeared before its lookback"


@pytest.mark.parametrize("name", ["ret_5", "ret_21", "mom_12_1", "beta_60", "vol_21"])
def test_key_features_registered(name):
    assert name in all_features()
