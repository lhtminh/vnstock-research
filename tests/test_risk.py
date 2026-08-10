"""Exposure-control tests.

The failure mode to guard against is a risk rule that peeks. A filter allowed to
see today's close would sidestep every crash perfectly and prove nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vnresearch.backtest import risk


def _level(returns) -> pd.Series:
    idx = pd.DatetimeIndex(pd.bdate_range("2010-01-01", periods=len(returns)))
    return pd.Series(100 * np.cumprod(1 + np.asarray(returns)), index=idx)


def test_trend_filter_is_point_in_time():
    """A crash on day t must not reduce exposure ON day t.

    The rule compares yesterday's close to yesterday's average, so the first
    risk-off day can be the day AFTER the break at the earliest — never the
    break itself.
    """
    rng = np.random.default_rng(0)
    up = rng.normal(0.002, 0.005, 300)
    crash = np.full(20, -0.05)
    lvl = _level(np.concatenate([up, crash]))

    e = risk.trend_exposure(lvl, window=200, risk_off=0.0)
    first_crash_day = lvl.index[300]
    assert e.loc[first_crash_day] == 1.0, "exposure reacted on the crash day itself"


def test_trend_filter_goes_risk_off_in_a_sustained_fall():
    rng = np.random.default_rng(1)
    lvl = _level(np.concatenate([rng.normal(0.002, 0.005, 300), np.full(60, -0.02)]))
    e = risk.trend_exposure(lvl, window=200, risk_off=0.0)
    assert e.iloc[-1] == 0.0, "still fully invested after a 60-day decline"


def test_vol_target_cuts_exposure_when_volatility_rises():
    rng = np.random.default_rng(2)
    calm = rng.normal(0.0003, 0.004, 300)
    wild = rng.normal(0.0, 0.045, 120)
    rets = pd.Series(
        np.concatenate([calm, wild]),
        index=pd.DatetimeIndex(pd.bdate_range("2010-01-01", periods=420)),
    )
    e = risk.vol_target_exposure(rets, target_vol=0.20, window=60)
    assert e.iloc[290] > e.iloc[-1], "exposure did not fall as volatility rose"
    assert e.iloc[-1] < 0.5


def test_exposure_never_levers_up():
    """Calm markets must not push exposure above 1. Surviving a crash is the
    question here; leverage in the quiet years is a different bet."""
    rng = np.random.default_rng(3)
    rets = pd.Series(
        rng.normal(0.0002, 0.001, 400),  # very low volatility
        index=pd.DatetimeIndex(pd.bdate_range("2010-01-01", periods=400)),
    )
    e = risk.vol_target_exposure(rets, target_vol=0.20, window=60)
    assert e.max() <= 1.0


def test_combined_takes_the_more_cautious_signal():
    """Either signal alone must be able to take risk off.

    Multiplying would let a calm tape cancel a downtrend. 2008 was both, but
    they do not always arrive together.
    """
    rng = np.random.default_rng(4)
    # A slow grind down: low volatility, but firmly below the average.
    lvl = _level(np.concatenate([rng.normal(0.002, 0.004, 300), np.full(80, -0.004)]))
    both = risk.combined_exposure(lvl, target_vol=0.20, trend_window=200, risk_off=0.0)
    vol_only = risk.combined_exposure(lvl, target_vol=0.20, trend_window=None)
    assert both.iloc[-1] < vol_only.iloc[-1]


def test_starts_fully_invested_before_the_window_fills():
    """No estimate yet must mean invested, not flat — otherwise the rule takes
    credit for sitting out a period it had no opinion about."""
    rng = np.random.default_rng(5)
    lvl = _level(rng.normal(0.001, 0.01, 300))
    e = risk.combined_exposure(lvl, target_vol=0.20, trend_window=200)
    assert e.iloc[0] == pytest.approx(1.0)
