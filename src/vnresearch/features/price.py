"""Volatility and intraday shape."""

from __future__ import annotations

from vnresearch.features.registry import lag, register, win, windows

_CAT = "price"
_CFG = windows()

for _n in _CFG["vol_windows"]:
    register(f"vol_{_n}", f"STDDEV_SAMP(ret) {win(_n)}", _CAT, _n)

# Downside deviation: only losing days count. Two stocks can share a volatility
# number while one of them got there entirely on the way down.
#
# The CASE is not decoration. `LEAST(NULL, 0)` returns 0 in DuckDB, not NULL, so
# the bare form fed AVG a real zero for every missing return and counted it as a
# genuinely calm day. panel.py nulls `ret` on suspect_move / band_anomaly /
# suspect_ohlc bars, which made the defect-flagged sessions precisely the ones
# read as calm. Same trap as invariant 3 and as idio_vol_60, which guards it.
register(
    "downside_vol_21",
    f"SQRT(AVG(CASE WHEN ret IS NOT NULL THEN POWER(LEAST(ret, 0), 2) END) {win(21)})",
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

# --- Range-based volatility -------------------------------------------------
#
# vol_21 above uses closes only, throwing away the high and the low — which is
# most of what the session told us. A stock that swung 8% intraday and closed
# flat reads as a calm day to a close-to-close estimator.
#
# Parkinson uses the range and is roughly 5x more efficient than close-to-close
# at the same window length; Garman-Klass adds the open and close and is better
# still. Both assume continuous trading and no gaps, which price limits and
# lunch breaks violate here, so they are estimates rather than truth — but they
# are estimates built from strictly more information.
#
# Expect these to correlate strongly with vol_21/vol_63. That is the point: if
# a better estimator dominates the one it duplicates, the weaker one should go.
_vol_w = _CFG["vol_windows"][0]

register(
    f"parkinson_{_vol_w}",
    f"""SQRT(AVG(POWER(LN(high / NULLIF(low, 0)), 2)) {win(_vol_w)}
             / (4 * LN(2)))""",
    _CAT,
    _vol_w,
)

# Garman-Klass: 0.5*ln(H/L)^2 - (2*ln2 - 1)*ln(C/O)^2, averaged then rooted.
# GREATEST guards the root, and the outer CASE guards GREATEST — on its own
# GREATEST(NULL, 0) returns 0, which would report a stock with no data as
# having exactly zero volatility. Same trap as idio_vol_60.
_gk = f"""AVG(0.5 * POWER(LN(high / NULLIF(low, 0)), 2)
              - (2 * LN(2) - 1) * POWER(LN(close / NULLIF(open, 0)), 2)) {win(_vol_w)}"""
register(
    f"garman_klass_{_vol_w}",
    f"CASE WHEN ({_gk}) IS NOT NULL THEN SQRT(GREATEST({_gk}, 0)) END",
    _CAT,
    _vol_w,
)

# Average true range, as a fraction of price. True range includes the overnight
# gap, so unlike hl_range it does not read a limit-gap open as a quiet session.
#
# `prev_close` from the panel rather than LAG(close, 1): SQL forbids a window
# function inside another window function's argument, and this needs the lag
# INSIDE an AVG. The panel column is the same value computed one layer down.
_atr_w = _CFG["atr_window"]
_tr = "GREATEST(high - low, ABS(high - prev_close), ABS(low - prev_close))"
register(
    f"atr_{_atr_w}_norm",
    f"AVG({_tr}) {win(_atr_w)} / NULLIF(close, 0)",
    _CAT,
    _atr_w + 1,
)

# Is volatility rising or falling? The level says how risky the name is; the
# ratio says whether something is happening to it right now.
_vs, _vl = _CFG["vol_windows"][0], _CFG["vol_windows"][1]
register(
    f"vol_ratio_{_vs}_{_vl}",
    f"STDDEV_SAMP(ret) {win(_vs)} / NULLIF(STDDEV_SAMP(ret) {win(_vl)}, 0)",
    _CAT,
    _vl,
)

# --- Overnight versus intraday ----------------------------------------------
#
# `gap` is the overnight half of the day's return; this is the other half. They
# are worth separating because they are earned by different people: the
# overnight move prices news nobody could trade on, the intraday move is the
# session's actual auction.
register("intraday_ret", "close / NULLIF(open, 0) - 1", _CAT, 1)

# What share of recent movement happened overnight. A name whose return arrives
# almost entirely at the open is one retail cannot actually capture.
#
# ABS in both places: signed sums cancel and would divide a small number by
# another small number for any name that went nowhere.
_range_w = _CFG["shape_window"]
register(
    f"overnight_frac_{_range_w}",
    f"""SUM(ABS(open / NULLIF(prev_close, 0) - 1)) {win(_range_w)}
        / NULLIF(SUM(ABS(ret)) {win(_range_w)}, 0)""",
    _CAT,
    _range_w + 1,
)
