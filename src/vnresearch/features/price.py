"""Volatility and intraday shape."""

from __future__ import annotations

from vnresearch.features.registry import lag, register, win

_CAT = "price"

for _n in (21, 63):
    register(f"vol_{_n}", f"STDDEV_SAMP(ret) {win(_n)}", _CAT, _n)

# Downside deviation: only losing days count. Two stocks can share a volatility
# number while one of them got there entirely on the way down.
register(
    "downside_vol_21",
    f"SQRT(AVG(POWER(LEAST(ret, 0), 2)) {win(21)})",
    _CAT,
    21,
)

# Overnight gap. In a market with price limits a big gap often means the
# session opened locked, so this doubles as a stress indicator.
register("gap", f"open / NULLIF({lag('close', 1)}, 0) - 1", _CAT, 2)

# Intraday range as a fraction of price.
register("hl_range", "(high - low) / NULLIF(close, 0)", _CAT, 1)

# Where in the day's range the close landed: 1 = closed on the high.
register(
    "close_location",
    "(close - low) / NULLIF(high - low, 0)",
    _CAT,
    1,
)

# How often the ticker actually traded recently. A name that prints a price
# every day is a different animal from one that trades twice a week, and the
# raw ADTV number hides that.
register(
    "trade_freq_21",
    f"AVG(CASE WHEN volume > 0 THEN 1.0 ELSE 0.0 END) {win(21)}",
    _CAT,
    21,
)
