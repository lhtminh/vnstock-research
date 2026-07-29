"""Momentum and reversal.

Momentum is the most reliable cross-sectional signal in most equity markets,
including Vietnam. Short-horizon returns usually behave the opposite way
(reversal), which is why 1- and 5-session returns are registered separately
rather than lumped in with the long ones.
"""

from __future__ import annotations

from vnresearch.features.registry import lag, register, win

_CAT = "momentum"

for _n in (1, 5, 21, 63, 126, 252):
    register(
        f"ret_{_n}",
        f"close / NULLIF({lag('close', _n)}, 0) - 1",
        _CAT,
        _n,
    )

# 12-month momentum skipping the most recent month. The skip matters: the last
# month is dominated by short-term reversal, which points the other way and
# cancels much of the signal if left in.
register(
    "mom_12_1",
    f"{lag('close', 21)} / NULLIF({lag('close', 252)}, 0) - 1",
    _CAT,
    252,
)

# Where the price sits in its own yearly range. Near the 52-week high is a
# distinct signal from raw momentum — it is bounded, so it does not blow up on
# a stock that tripled.
register(
    "dist_52w_high",
    f"close / NULLIF(MAX(close) {win(252)}, 0) - 1",
    _CAT,
    252,
)
register(
    "dist_52w_low",
    f"close / NULLIF(MIN(close) {win(252)}, 0) - 1",
    _CAT,
    252,
)
