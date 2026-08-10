"""Market-relative features, against VNINDEX.

These are NULL before 2004-01-05, where index_series begins, while bars go back
to 2002. That is a data boundary, not a bug — do not fill it. (It was
2019-09-12 before the index history was extended; the sample starts at 2009 for
a different reason, which is that earlier years have no cross-section to rank.)
"""

from __future__ import annotations

from vnresearch.features.registry import register, win, windows

_CAT = "market"
_CFG = windows()
_W = _CFG["beta_window"]
_WL = _CFG["beta_window_long"]

register("beta_60", f"REGR_SLOPE(ret, mkt_ret) {win(_W)}", _CAT, _W)
register("alpha_60", f"REGR_INTERCEPT(ret, mkt_ret) {win(_W)}", _CAT, _W)

# The same beta over a slower window. On its own it says little that beta_60
# does not; as a PAIR the two say whether a name's market sensitivity is stable,
# and an unstable beta makes the residual target noisier for that stock.
register(f"beta_{_WL}", f"REGR_SLOPE(ret, mkt_ret) {win(_WL)}", _CAT, _WL)

# Correlation with the market, signed. Not a restatement of beta: beta is
# correlation scaled by the volatility ratio, so a low-vol stock that tracks the
# index perfectly has a low beta and a correlation near 1. Which of those a
# model wants depends on what else it is holding.
register(f"mkt_corr_{_W}", f"CORR(ret, mkt_ret) {win(_W)}", _CAT, _W)

# Idiosyncratic volatility: the part of the move the market does not explain.
# Derived from R^2 rather than by regressing residuals, which SQL cannot do in
# one window pass.
#
# The outer CASE is load-bearing. GREATEST(1 - NULL, 0) returns 0 in DuckDB
# rather than NULL, so without it every pre-2019 row — where there is no index
# to regress against — gets a fabricated idiosyncratic volatility of exactly
# zero, which a tree will happily split on.
register(
    "idio_vol_60",
    f"""CASE WHEN REGR_R2(ret, mkt_ret) {win(_W)} IS NOT NULL
             THEN STDDEV_SAMP(ret) {win(_W)}
                  * SQRT(GREATEST(1 - REGR_R2(ret, mkt_ret) {win(_W)}, 0))
        END""",
    _CAT,
    _W,
)

# Return in excess of the market over the same window — the simplest form of
# "did this name beat the index", and largely free of the market's own trend.
for _n in (5, 21, 63):
    register(
        f"excess_ret_{_n}",
        f"(EXP(SUM(LN(1 + ret)) {win(_n)}) - 1) - (EXP(SUM(LN(1 + mkt_ret)) {win(_n)}) - 1)",
        _CAT,
        _n,
    )
