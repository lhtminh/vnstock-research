"""Exposure control: how much of the book to have on at all.

WHY THIS AND NOT MORE FEATURES. The 2008 stress test scored IC +0.0596 — the
model ranked correctly through the crash — and the strategy still lost 76%
against the index's 66%, drawing down 84%. Prediction was never the problem. A
long-only book that is always 100% invested loses roughly the market in a -66%
year no matter how well it sorts the survivors. The fix has to reduce EXPOSURE,
not improve the ordering.

WHY NOT JUST HEDGE. VN30 futures are the only shortable instrument in Vietnam
and they launched in 2017; the VN30 index itself only starts 2012. There was
nothing to hedge 2008 with. A crisis rule that works here has to be able to sit
in cash.

TWO SIGNALS, BOTH POINT-IN-TIME AND BOTH SLOW.

  volatility target   scale exposure by target/realised. Volatility rises
                      BEFORE and during a crash, not after, so this de-risks
                      into falling markets without predicting anything.

  trend filter        cut exposure while the index is below its own long
                      moving average. Crude, old, and the most reliably
                      documented crisis mitigant there is.

Both are deliberately lagged by a day: the exposure used on date t is computed
from data through t-1, so nothing here can see the session it is sizing.

NOTHING IS FREE. Either rule costs return in a rising market — that is the
premium being paid for the protection, and the honest way to present it is to
report the calm-period cost next to the crisis benefit, never one alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def realised_vol(returns: pd.Series, window: int = 60) -> pd.Series:
    """Trailing annualised volatility, shifted so date t uses data through t-1."""
    return returns.rolling(window, min_periods=window // 2).std().shift(1) * np.sqrt(TRADING_DAYS)


def vol_target_exposure(
    bench_returns: pd.Series,
    target_vol: float = 0.20,
    window: int = 60,
    max_exposure: float = 1.0,
    min_exposure: float = 0.0,
) -> pd.Series:
    """Exposure that holds portfolio volatility near target.

    Capped at max_exposure rather than allowed to lever up in calm markets: the
    question here is surviving a crash, and leverage in the quiet years is a
    different bet that would flatter the result for unrelated reasons.
    """
    vol = realised_vol(bench_returns, window)
    exposure = (target_vol / vol).clip(lower=min_exposure, upper=max_exposure)
    # Before the window fills there is no estimate. Start fully invested rather
    # than flat, so the rule cannot claim credit for sitting out the beginning.
    return exposure.fillna(max_exposure)


def trend_exposure(
    bench_level: pd.Series,
    window: int = 200,
    risk_off: float = 0.0,
    risk_on: float = 1.0,
) -> pd.Series:
    """Full exposure while the index is above its moving average, risk_off below.

    The average and the comparison are both shifted: the decision for date t is
    made on t-1's close against t-1's average.
    """
    ma = bench_level.rolling(window, min_periods=window // 2).mean()
    above = (bench_level > ma).shift(1)
    return pd.Series(np.where(above.fillna(True), risk_on, risk_off), index=bench_level.index)


def combined_exposure(
    bench_level: pd.Series,
    target_vol: float | None = 0.20,
    vol_window: int = 60,
    trend_window: int | None = 200,
    risk_off: float = 0.0,
    max_exposure: float = 1.0,
) -> pd.Series:
    """Both rules together, taking the MORE cautious of the two.

    Multiplying them would let a calm market cancel a downtrend and vice versa.
    The minimum means either signal alone can take risk off, which is the point:
    2008 was both high-volatility and a downtrend, but the two do not always
    arrive together.
    """
    parts = []
    rets = bench_level.pct_change()
    if target_vol:
        parts.append(vol_target_exposure(rets, target_vol, vol_window, max_exposure))
    if trend_window:
        parts.append(trend_exposure(bench_level, trend_window, risk_off, max_exposure))
    if not parts:
        return pd.Series(max_exposure, index=bench_level.index)
    return pd.concat(parts, axis=1).min(axis=1).clip(0.0, max_exposure)
