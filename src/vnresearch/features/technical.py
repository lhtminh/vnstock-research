"""Chart indicators: RSI, MACD, Bollinger, moving-average ratios, stochastic.

WHY THESE ARE SIMPLE MOVING AVERAGES AND NOT EMA/WILDER.

The textbook forms of RSI and MACD smooth exponentially, which is recursive:
today's value depends on yesterday's value and so on back to the first bar. That
cannot be written as a bounded `ROWS BETWEEN n PRECEDING AND CURRENT ROW` frame,
and `win()` — the only frame builder here — deliberately cannot express anything
else. Bypassing it would also cost the guarantee that
`test_features_are_truncation_invariant` gives us.

The substitution is cheap. Over the same span an SMA and an EMA of a price
series correlate above ~0.95, and this model consumes CROSS-SECTIONAL RANKS of
the value rather than the value itself, so the small level differences mostly
disappear in the ranking. Matching a charting package exactly is not worth
giving up the one test that proves no feature looks forward.

EVERYTHING IS NORMALISED BY PRICE. A raw MACD is in VND, so the same reading
means something different on a 10,000 VND stock and a 200,000 VND one. A
cross-sectional model would be ranking price levels with extra steps.
"""

from __future__ import annotations

from vnresearch.features.registry import register, win, windows

_CAT = "technical"
_CFG = windows()

_RSI = _CFG["rsi_window"]
_FAST, _SLOW = _CFG["macd_fast"], _CFG["macd_slow"]
_BB = _CFG["bollinger_window"]
_STOCH = _CFG["stoch_window"]

# Relative strength index, 0..100. Above 70 is conventionally "overbought".
#
# The CASE around each side is the invariant-3 guard and it is not optional:
# `GREATEST(NULL, 0)` returns 0 in DuckDB, so a missing return would be counted
# as a real day with zero gain AND zero loss, dragging the ratio toward 1 and
# the index toward 50 — a confident "neutral" reading manufactured from absent
# data. panel.py nulls `ret` on every defect-flagged bar, so this would fire
# precisely where the data is worst.
_gain = f"AVG(CASE WHEN ret IS NOT NULL THEN GREATEST(ret, 0) END) {win(_RSI)}"
_loss = f"AVG(CASE WHEN ret IS NOT NULL THEN GREATEST(-ret, 0) END) {win(_RSI)}"
register(
    f"rsi_{_RSI}",
    f"100 - 100 / (1 + {_gain} / NULLIF({_loss}, 0))",
    _CAT,
    _RSI,
)

# MACD line: the gap between a fast and a slow average, as a fraction of price.
# Positive means the recent trend is above the older one.
register(
    f"ma_gap_{_FAST}_{_SLOW}",
    f"(AVG(close) {win(_FAST)} - AVG(close) {win(_SLOW)}) / NULLIF(close, 0)",
    _CAT,
    _SLOW,
)

# Bollinger %B: where the close sits inside its own band, in standard
# deviations. Already scale-free, so no price normalisation is needed.
register(
    f"bb_pct_{_BB}",
    f"(close - AVG(close) {win(_BB)}) / NULLIF(2 * STDDEV_SAMP(close) {win(_BB)}, 0)",
    _CAT,
    _BB,
)

# Bandwidth: how wide the band is relative to the average. This is a volatility
# reading taken from prices rather than returns, and it is the one that
# "squeeze" strategies watch.
register(
    f"bb_width_{_BB}",
    f"(4 * STDDEV_SAMP(close) {win(_BB)}) / NULLIF(AVG(close) {win(_BB)}, 0)",
    _CAT,
    _BB,
)

# Distance from the classic long moving averages. Bounded below at -1 and
# unbounded above, like dist_52w_high, which is why it is expressed as a ratio
# rather than a crossover flag: a boolean throws away how far.
for _n in _CFG["ma_windows"]:
    register(
        f"ma_ratio_{_n}",
        f"close / NULLIF(AVG(close) {win(_n)}, 0) - 1",
        _CAT,
        _n,
    )

# Stochastic %K: position within the recent high-low range. This is
# close_location generalised from one session to many — that one says where the
# close sat inside TODAY's bar, this says where it sits in a fortnight of them.
register(
    f"stoch_{_STOCH}",
    f"""(close - MIN(low) {win(_STOCH)})
        / NULLIF(MAX(high) {win(_STOCH)} - MIN(low) {win(_STOCH)}, 0)""",
    _CAT,
    _STOCH,
)
