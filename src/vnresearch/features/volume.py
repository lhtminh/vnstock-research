"""Liquidity and volume.

Turnover here is the close*volume proxy from the panel — the source feed does
not populate matched value. It overstates limit-locked sessions and ignores
put-through trades entirely.
"""

from __future__ import annotations

from vnresearch.features.registry import register, win

_CAT = "volume"

# Size proxy until company_snapshot covers more than 51 tickers. Log because
# turnover spans six orders of magnitude across this universe.
register("log_adtv", "LN(NULLIF(adtv, 0))", _CAT, 20)

# Today's turnover against its own recent average: a volume spike.
register("turnover_ratio", "turnover / NULLIF(adtv, 0)", _CAT, 20)

for _n in (5, 21):
    register(
        f"turnover_ratio_{_n}",
        f"AVG(turnover) {win(_n)} / NULLIF(adtv, 0)",
        _CAT,
        max(_n, 20),
    )

# Amihud illiquidity: price impact per unit of turnover. High means the stock
# moves a lot on little volume, which is exactly where backtest returns are
# most likely to be unreachable in practice.
register(
    "amihud_21",
    f"AVG(ABS(ret) / NULLIF(turnover, 0)) {win(21)} * 1e9",
    _CAT,
    21,
)

# Share of recent sessions that were limit-locked or had no trade. A name that
# is frequently untradeable should be treated differently by the model even
# when today's bar looks normal.
register(
    "untradeable_frac_21",
    f"AVG(CASE WHEN tradeable THEN 0.0 ELSE 1.0 END) {win(21)}",
    _CAT,
    21,
)
