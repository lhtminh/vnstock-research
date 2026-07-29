"""Market-relative features, against VNINDEX.

These are NULL before 2019-09-12 because index_series starts there, while bars
go back to 2002. That is a data boundary, not a bug — do not fill it.
"""

from __future__ import annotations

from vnresearch.features.registry import register, win

_CAT = "market"
_W = 60

register("beta_60", f"REGR_SLOPE(ret, mkt_ret) {win(_W)}", _CAT, _W)
register("alpha_60", f"REGR_INTERCEPT(ret, mkt_ret) {win(_W)}", _CAT, _W)

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
