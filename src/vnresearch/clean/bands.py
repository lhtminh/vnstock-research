"""Vietnamese price limits and tick sizes.

daily_prices.bar_status is NULL for all 4.77M rows and ceiling/floor are never
populated, so a limit-locked session is indistinguishable from a normal one
until we rebuild the bands here. That matters because you cannot buy on a
limit-up day — the offer side is empty — and filling on those days is the
easiest way to invent returns that were never available.

Two accuracy limits, both deliberate:

1. `symbols.exchange` is the CURRENT venue only. ACB traded on HNX until Dec
   2020, so a band picked from today's exchange is wrong for its old bars.
2. The stored prices are ADJUSTED, so historical bars are not tick-aligned —
   a cash dividend multiplies the whole series by a non-round factor. Tick
   reasoning therefore degrades the further back you go.

Both are handled by classifying conservatively: anything we cannot judge gets
`unknown`, never a guessed limit.
"""

from __future__ import annotations

# Daily price limit by exchange, as a fraction of the reference price.
BANDS = {"HOSE": 0.07, "HNX": 0.10, "UPCOM": 0.15}
DEFAULT_BAND = 0.15  # unknown exchange: assume the widest, so we under-claim limits

# Cross-exchange threshold for "this move is too big to be legal anywhere".
# Used for data-quality flags only, never to classify a limit.
MAX_LEGAL_MOVE = 0.155

# Below this price one minimum tick is a large fraction of the band, so the
# limit test stops being meaningful. Matches vn-audit's exclusion.
MIN_PRICE_FOR_BAND = 1000.0

# A reference price more than this many calendar days old is stale — the
# ticker was suspended or newly resumed, and the band was reset.
MAX_REF_GAP_DAYS = 4

# How far a limit close may sit from the nominal band, as a return.
#
# The floor exists because the stored prices are ADJUSTED and therefore not
# tick-aligned, so a genuine limit-up can compute to 6.9% instead of 7.0%.
# Missing it would label the bar `normal` and let a backtest fill on a day with
# no offer side — the expensive mistake. Calling a 6.8% move a limit merely
# skips a trade, so the errors are worth trading in this direction.
#
# The cap stops the tolerance from swallowing the band on cheap stocks.
MIN_TOLERANCE = 0.003
MAX_TOLERANCE = 0.02


def band(exchange: str | None) -> float:
    """Price limit for an exchange."""
    return BANDS.get(exchange or "", DEFAULT_BAND)


def tick_size(price: float, exchange: str | None) -> int:
    """Minimum price increment in VND.

    HOSE steps by price; HNX and UPCOM are a flat 100.
    """
    if exchange == "HOSE":
        if price < 10_000:
            return 10
        if price < 50_000:
            return 50
        return 100
    return 100


def ceiling_price(ref: float, exchange: str | None) -> float:
    """Highest legal price for the session, rounded down to a tick."""
    tick = tick_size(ref, exchange)
    raw = ref * (1 + band(exchange))
    return max(int(raw / tick) * tick, ref + tick)


def floor_price(ref: float, exchange: str | None) -> float:
    """Lowest legal price for the session, rounded up to a tick."""
    tick = tick_size(ref, exchange)
    raw = ref * (1 - band(exchange))
    stepped = -(-raw // tick) * tick  # ceil division
    return min(float(stepped), max(ref - tick, tick))


def tolerance(ref: float, exchange: str | None) -> float:
    """How far below the nominal band a limit close can land, as a return.

    The ceiling is the band rounded down to a tick, so the return at ceiling is
    at least `band - tick/ref`. Clamped at both ends — see the constants.
    """
    return min(max(tick_size(ref, exchange) / ref, MIN_TOLERANCE), MAX_TOLERANCE)
