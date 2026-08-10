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
            op = px * (1 + rng.normal(0, 0.002))
            # A varying range, bracketing open and close so high/low are
            # coherent. A constant 1% band would leave every range-based
            # volatility estimator reporting the same number on every row, and
            # a test cannot see the difference between that and a broken one.
            half = 0.005 + 0.02 * rng.random()
            # Mostly normal, with occasional band-locked sessions so the
            # limit_*_frac features have something to count.
            status = "normal"
            if (t + i) % 47 == 0:
                status = "limit_up"
            elif (t + i) % 53 == 0:
                status = "limit_down"
            rows.append(
                {
                    "ticker": f"T{t}",
                    "date": d,
                    "open": op,
                    "high": max(px, op) * (1 + half),
                    "low": min(px, op) * (1 - half),
                    "close": px,
                    "volume": vol,
                    "exchange": "HOSE",
                    "symbol_status": "active",
                    "bar_status": status,
                    "prev_close": prev,
                    # The session's price limit. NULL on a few rows, as the
                    # cleaning layer emits when the operative band is unknown,
                    # so band_proximity is exercised on both paths.
                    "band_pct": None if i % 97 == 0 else 0.07,
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


# ---------------------------------------------------------------------------
# Formulas pinned to known values.
#
# Truncation invariance proves a feature does not look forward. It says nothing
# about whether the arithmetic is right — a formula that is wrong in a stable
# way passes it happily. RSI, ATR and the range-based volatility estimators all
# have off-by-one and which-price-goes-where mistakes available to them, so the
# ones below are checked against a value computed a different way.
# ---------------------------------------------------------------------------


def _known_panel(
    tmp_path,
    *,
    closes,
    opens=None,
    highs=None,
    lows=None,
    rets="derive",
    bar_status=None,
    band_pct=0.07,
    name="panel_known.parquet",
) -> str:
    """A one-ticker panel with prices stated exactly, for pinning arithmetic.

    `rets="derive"` computes close-to-close returns; passing a list instead lets
    a test inject the NULLs that panel.py writes on defect-flagged bars, which
    is the case the DuckDB LEAST/GREATEST trap hides in.
    """
    n = len(closes)
    closes = [float(c) for c in closes]
    opens = [float(o) for o in (opens if opens is not None else closes)]
    highs = [
        float(h)
        for h in (highs if highs is not None else [max(c, o) for c, o in zip(closes, opens)])
    ]
    lows = [
        float(low)
        for low in (lows if lows is not None else [min(c, o) for c, o in zip(closes, opens)])
    ]
    prev = [None] + closes[:-1]
    if rets == "derive":
        rets = [None if p is None else closes[i] / p - 1 for i, p in enumerate(prev)]
    dates = pd.bdate_range("2023-01-02", periods=n).date
    rows = [
        {
            "ticker": "T0",
            "date": dates[i],
            "open": opens[i],
            "high": highs[i],
            "low": lows[i],
            "close": closes[i],
            "volume": 100_000,
            "exchange": "HOSE",
            "symbol_status": "active",
            "bar_status": (bar_status[i] if bar_status else "normal"),
            "prev_close": prev[i],
            "ret": rets[i],
            "turnover": closes[i] * 100_000,
            "adtv": closes[i] * 100_000,
            "adtv_obs": 20,
            "tradeable": True,
            "in_universe": True,
            "mkt_ret": 0.001,
            "band_pct": band_pct,
            "dilution_252": 0.0,
            "seas_month": 0.0,
            "seas_tet": 0.0,
            "days_from_tet": 10,
            "peer_ret_1": 0.0,
            "peer_ret_5": 0.0,
            "peer_ret_21": 0.0,
            "peer_corr": 0.3,
            "peer_n": 10,
        }
        for i in range(n)
    ]
    path = tmp_path / name
    pd.DataFrame(rows).to_parquet(path)
    return path.as_posix()


def test_a_null_return_is_skipped_not_counted_as_zero(tmp_path):
    """The trap that made downside_vol_21 wrong, pinned so it cannot come back.

    `LEAST(NULL, 0)` and `GREATEST(NULL, 0)` both return 0 in DuckDB rather than
    NULL, so a bare form hands AVG a real zero for every missing return and
    counts it as a genuinely calm — or genuinely flat — session. panel.py nulls
    `ret` on suspect_move / band_anomaly / suspect_ohlc bars, which makes the
    defect-flagged sessions exactly the ones that would be read as calm.
    """
    n = 30
    closes = [10_000.0 * (0.99**i) for i in range(n)]  # every return is -1%
    rets = [None if i == 0 else -0.01 for i in range(n)]
    for i in (25, 26, 27, 28, 29):  # five nulled bars inside the last window
        rets[i] = None
    df = _compute(_known_panel(tmp_path, closes=closes, rets=rets))
    last = df.sort_values("date").iloc[-1]

    # 16 real losing days of -1% in the trailing 21, five NULL. Skipping the
    # NULLs gives exactly 0.01; counting them as zero gives 0.01*sqrt(16/21).
    assert last["downside_vol_21"] == pytest.approx(0.01, rel=1e-9)
    assert last["downside_vol_21"] != pytest.approx(0.01 * np.sqrt(16 / 21), rel=1e-6)

    # Same trap in an average of an indicator: a NULL falling through to ELSE
    # would be counted as a real "not up" day and shrink the fraction.
    rets2 = [None if i == 0 else (0.02 if i % 3 == 0 else -0.01) for i in range(n)]
    rets2[27] = rets2[28] = None
    df2 = _compute(_known_panel(tmp_path, closes=closes, rets=rets2, name="p2.parquet"))
    last2 = df2.sort_values("date").iloc[-1]
    real = [r for r in rets2[-21:] if r is not None]
    assert last2["up_day_frac_21"] == pytest.approx(sum(r > 0 for r in real) / len(real), rel=1e-9)
    assert last2["up_day_frac_21"] != pytest.approx(sum(r > 0 for r in real) / 21, rel=1e-6), (
        "NULL days were counted as down days"
    )

    # RSI is the interesting non-case, and worth recording so nobody 'fixes' the
    # guard away as pointless OR expects it to move the number. Both averages
    # would gain the same inflated denominator, so the ratio between them — and
    # therefore the index — is unchanged. The guard stays because it is correct
    # and because the expression would stop being safe the moment either side
    # is used on its own.
    real14 = [r for r in rets2[-14:] if r is not None]
    assert last2["rsi_14"] == pytest.approx(
        100
        - 100 / (1 + np.mean([max(r, 0) for r in real14]) / np.mean([max(-r, 0) for r in real14])),
        rel=1e-9,
    )


def test_parkinson_matches_the_closed_form(tmp_path):
    """sqrt(mean(ln(H/L)^2) / (4 ln 2)) on a constant range."""
    n = 25
    closes = [10_000.0] * n
    highs = [10_100.0] * n
    lows = [9_900.0] * n
    df = _compute(_known_panel(tmp_path, closes=closes, highs=highs, lows=lows))
    last = df.sort_values("date").iloc[-1]
    expected = np.sqrt(np.log(10_100 / 9_900) ** 2 / (4 * np.log(2)))
    assert last["parkinson_21"] == pytest.approx(expected, rel=1e-9)


def test_atr_uses_the_previous_close_not_just_the_bar(tmp_path):
    """True range includes the overnight gap — that is what separates it from
    hl_range, and getting it wrong makes a limit-gap open look like a calm day."""
    n = 20
    # Each bar gaps up 100 from the prior close and then trades in a 50 band.
    closes = [10_000.0 + 100 * i for i in range(n)]
    opens = [c for c in closes]
    highs = [c + 25 for c in closes]
    lows = [c - 25 for c in closes]
    df = _compute(_known_panel(tmp_path, closes=closes, opens=opens, highs=highs, lows=lows))
    last = df.sort_values("date").iloc[-1]
    # TR = max(H-L, |H-prev|, |L-prev|) = max(50, 125, 75) = 125 on every bar
    # after the first, so a 14-session average is exactly 125.
    assert last["atr_14_norm"] == pytest.approx(125.0 / closes[-1], rel=1e-9)


def test_stoch_is_the_position_in_the_window_range(tmp_path):
    """Closing at the top of a 14-session range is 1.0, at the bottom 0.0."""
    n = 20
    closes = [10_000.0] * (n - 1) + [10_500.0]
    highs = [10_500.0] * n
    lows = [9_500.0] * n
    df = _compute(_known_panel(tmp_path, closes=closes, highs=highs, lows=lows))
    rows = df.sort_values("date")
    assert rows.iloc[-1]["stoch_14"] == pytest.approx(1.0, rel=1e-9)
    # A mid-range close: (10000 - 9500) / (10500 - 9500) = 0.5
    assert rows.iloc[-2]["stoch_14"] == pytest.approx(0.5, rel=1e-9)


def test_ma_ratio_is_close_over_its_own_average(tmp_path):
    n = 60
    closes = [10_000.0] * (n - 1) + [11_000.0]
    df = _compute(_known_panel(tmp_path, closes=closes))
    last = df.sort_values("date").iloc[-1]
    expected = 11_000.0 / ((10_000.0 * 49 + 11_000.0) / 50) - 1
    assert last["ma_ratio_50"] == pytest.approx(expected, rel=1e-9)


def test_band_proximity_is_the_fraction_of_the_limit_used(tmp_path):
    """A ceiling close is 1.0; the value is NULL where the band is unknown."""
    closes = [10_000.0, 10_700.0, 10_400.0]  # +7% then -2.8% on a 7% band
    df = _compute(_known_panel(tmp_path, closes=closes)).sort_values("date")
    assert df.iloc[1]["band_proximity"] == pytest.approx(1.0, rel=1e-9)
    assert df.iloc[2]["band_proximity"] == pytest.approx((10_400 / 10_700 - 1) / 0.07, rel=1e-9)

    unknown = _compute(
        _known_panel(tmp_path, closes=closes, band_pct=None, name="p_nb.parquet")
    ).sort_values("date")
    assert unknown["band_proximity"].isna().all(), (
        "an unknown band must yield NULL, never a division by a guessed 7%"
    )


def test_limit_fractions_count_the_right_status(tmp_path):
    n = 70
    status = ["normal"] * n
    for i in (60, 62, 64):
        status[i] = "limit_up"
    for i in (61, 63):
        status[i] = "limit_down"
    df = _compute(_known_panel(tmp_path, closes=[10_000.0] * n, bar_status=status))
    last = df.sort_values("date").iloc[-1]
    assert last["limit_up_frac_63"] == pytest.approx(3 / 63, rel=1e-9)
    assert last["limit_down_frac_63"] == pytest.approx(2 / 63, rel=1e-9)
