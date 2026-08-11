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

from vnresearch.features.registry import TECHNICAL, register, win, windows

_CAT = TECHNICAL
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
    direction=-1,  # a volatility reading; wider is worse, not better
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

# --- volume-aware indicators -------------------------------------------------
#
# Everything above reads price alone. These ask whether the volume AGREED with
# the move, which is the same question the mentor's PVDI asks from the other
# end: PVDI measures the price-volume relationship breaking down, these measure
# its current sign and strength.
#
# All of them lean on `ret` rather than a lagged close, deliberately. A
# LAG(close) inside these aggregates would be a window function nested inside
# another window function, which SQL forbids; `ret` is that same comparison
# already made one layer down, in the panel.

# Money Flow Index — RSI weighted by traded value. A 10% rise on no volume and
# the same rise on ten times normal volume are identical to rsi_14. They are not
# the same event, and this is the one that separates them.
_tp = "(high + low + close) / 3"
_mf_pos = f"SUM(CASE WHEN ret > 0 THEN {_tp} * volume ELSE 0 END) {win(_RSI)}"
_mf_neg = f"SUM(CASE WHEN ret < 0 THEN {_tp} * volume ELSE 0 END) {win(_RSI)}"
register(
    f"mfi_{_RSI}",
    f"100 - 100 / (1 + {_mf_pos} / NULLIF({_mf_neg}, 0))",
    _CAT,
    _RSI,
)

# Chaikin Money Flow: where each session closed inside its own range, weighted
# by that session's volume. Positive means the volume arrived on days that
# closed strong — accumulation rather than distribution.
_sh = _CFG["shape_window"]
_clv = "((close - low) - (high - close)) / NULLIF(high - low, 0)"
register(
    f"cmf_{_sh}",
    f"SUM({_clv} * volume) {win(_sh)} / NULLIF(SUM(volume) {win(_sh)}, 0)",
    _CAT,
    _sh,
)

# On-balance volume as a SHARE rather than a running total. Textbook OBV is a
# cumulative sum from the first bar ever: unbounded, not comparable between two
# stocks, and not expressible in a bounded frame in any case. The idea survives
# over a fixed window — of the volume that traded, what share came on up days.
register(
    f"obv_frac_{_sh}",
    f"""SUM(CASE WHEN ret > 0 THEN volume WHEN ret < 0 THEN -volume ELSE 0 END) {win(_sh)}
        / NULLIF(SUM(volume) {win(_sh)}, 0)""",
    _CAT,
    _sh,
)

# Distance from the window's volume-weighted average price. Real order flow is
# anchored to VWAP because that is what execution gets measured against.
# `turnover` is the panel's close*volume proxy — the feed does not publish
# matched value — so this is a VWAP built from closes, not from every print.
register(
    f"vwap_gap_{_sh}",
    f"close / NULLIF(SUM(turnover) {win(_sh)} / NULLIF(SUM(volume) {win(_sh)}, 0), 0) - 1",
    _CAT,
    _sh,
)

# --- trend quality -----------------------------------------------------------

# Keltner position: the close inside a band built from average TRUE RANGE rather
# than standard deviation. Worth having next to bb_pct because the two disagree
# exactly where it matters — a stock that gaps every morning has a wide true
# range and a narrow close-to-close deviation, so Bollinger calls it calm and
# this does not. Two window aggregates side by side, neither nested in the other.
_tr = "GREATEST(high - low, ABS(high - prev_close), ABS(low - prev_close))"
register(
    f"keltner_pct_{_BB}",
    f"(close - AVG(close) {win(_BB)}) / NULLIF(2 * AVG({_tr}) {win(_BB)}, 0)",
    _CAT,
    _BB + 1,
)

# Drift per unit of the noise around it — the t-statistic of the trend. Two
# names can share a 21-day return while one walked there and the other arrived
# by accident, and only this one tells them apart.
register(
    f"trend_quality_{_sh}",
    f"AVG(ret) {win(_sh)} / NULLIF(STDDEV_SAMP(ret) {win(_sh)}, 0)",
    _CAT,
    _sh,
)

# Position in the yearly channel. dist_52w_high measures against the high alone,
# so a stock 10% off its high reads the same whether the low is 15% below or
# 300% below. This places the close between both ends.
_yr = _CFG["momentum_windows"][-1]
register(
    f"channel_pos_{_yr}",
    f"""(close - MIN(low) {win(_yr)})
        / NULLIF(MAX(high) {win(_yr)} - MIN(low) {win(_yr)}, 0)""",
    _CAT,
    _yr,
)

# A second, faster MACD pair. 12/26 is a monthly instrument; this one turns
# inside a fortnight, and the two disagreeing is what says a trend is
# accelerating or rolling over.
register(
    "ma_gap_5_20",
    f"(AVG(close) {win(5)} - AVG(close) {win(20)}) / NULLIF(close, 0)",
    _CAT,
    20,
)

# RSI at a second speed. 14 sessions is convention and nothing more. A
# 5-session reading turns fast enough to catch a squeeze — and at that horizon
# the reversal effect momentum.py documents dominates, so unlike rsi_14 a high
# reading here is a caution rather than strength.
_gain_f = f"AVG(CASE WHEN ret IS NOT NULL THEN GREATEST(ret, 0) END) {win(5)}"
_loss_f = f"AVG(CASE WHEN ret IS NOT NULL THEN GREATEST(-ret, 0) END) {win(5)}"
register("rsi_5", f"100 - 100 / (1 + {_gain_f} / NULLIF({_loss_f}, 0))", _CAT, 5, direction=-1)
