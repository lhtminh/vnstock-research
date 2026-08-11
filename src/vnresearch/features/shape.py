"""Return-distribution shape, and the limit-hit behaviour specific to Vietnam.

Two ideas share this module because they answer the same question — how
SPECULATIVE is this name — from two directions.

The distribution side is the standard one. A stock is not summarised by its
volatility: two names with identical vol differ enormously if one got there
through a single +12% day and the other drifted. Investors are known to overpay
for the first kind (the "lottery demand" or MAX effect), so the biggest single
day in the recent past predicts weak subsequent returns even after controlling
for volatility.

The limit side has no analogue in a US or European model. HOSE, HNX and UPCOM
cap the daily move at 7/10/15%, so a name under sustained speculative pressure
does not gap — it prints a run of ceiling closes, each one a session where the
offer side was empty. That run is a direct, countable signature of the pressure,
and `bar_status` has already identified every one of those bars for a completely
different purpose (deciding what a backtest may fill).
"""

from __future__ import annotations

from vnresearch.features.registry import SPECULATION, register, win, windows

_CAT = SPECULATION
_CFG = windows()
_W = _CFG["shape_window"]
_L = _CFG["limit_window"]

# The largest single-session return in the recent past. MAX skips NULLs, so no
# guard is needed here — unlike the AVG-based features below, where DuckDB's
# LEAST/GREATEST would turn a missing return into a real zero.
register(f"max_ret_{_W}", f"MAX(ret) {win(_W)}", _CAT, _W)

# Skewness of returns. Negative means the tail is on the downside. Distinct from
# max_ret: skew describes the whole shape, max_ret one observation.
register(f"skew_{_W}", f"SKEWNESS(ret) {win(_W)}", _CAT, _W)

# What fraction of recent sessions were up. This is momentum CONSISTENCY, and it
# separates a steady climber from a name that is flat except for two huge days —
# two stocks that can share an identical ret_21.
#
# Written so a NULL return yields NULL and is skipped by AVG, rather than
# falling into an ELSE and being counted as a down day.
register(
    f"up_day_frac_{_W}",
    f"AVG(CASE WHEN ret > 0 THEN 1.0 WHEN ret IS NOT NULL THEN 0.0 END) {win(_W)}",
    _CAT,
    _W,
    # The one feature in this dimension that is not a warning sign. Steady
    # participation is quality; the SPECULATION default would mark a stock down
    # for climbing consistently.
    direction=+1,
)

# How often this name closed pinned at the ceiling or the floor. bar_status is
# never NULL — bars.py's CASE always resolves — so a plain ELSE is safe here.
#
# These are deliberately NOT folded into one signed number. A run of ceilings and
# a run of floors are both "hit the limit a lot", but they are opposite states of
# the world, and averaging them would cancel a volatile name into a calm-looking
# one.
register(
    f"limit_up_frac_{_L}",
    f"AVG(CASE WHEN bar_status = 'limit_up' THEN 1.0 ELSE 0.0 END) {win(_L)}",
    _CAT,
    _L,
)
register(
    f"limit_down_frac_{_L}",
    f"AVG(CASE WHEN bar_status = 'limit_down' THEN 1.0 ELSE 0.0 END) {win(_L)}",
    _CAT,
    _L,
)

# Today's move as a fraction of the room the exchange allowed: 1.0 is a ceiling
# close, -1.0 a floor. This is what makes a return comparable across venues — a
# +9% day is unremarkable on UPCOM's 15% band and impossible on HOSE's 7%.
#
# band_pct is NULL wherever the operative limit could not be established (the
# 2008 emergency window, stale reference prices, sub-1000 VND stocks), so this
# inherits "unknown, never guessed" from the cleaning layer instead of dividing
# by a current-era band that did not apply.
register("band_proximity", "ret / NULLIF(band_pct, 0)", _CAT, 2)
