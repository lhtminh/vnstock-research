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

# Daily price limits, and they are NOT constant through history.
#
# On 2013-01-15 the regulator widened every band: HOSE 5% -> 7%, HNX 7% -> 10%,
# UPCOM 10% -> 15%. Applying today's numbers to older bars silently reclassifies
# every genuine limit-up as an ordinary session, and the backtest then fills on
# a day that had no offer side — exactly the error bar_status exists to prevent.
#
# Verified from the data rather than taken on trust. Positive HOSE returns pile
# up in the 4.5-5.0% bucket through 2012 (6.1% of all returns in 2010) and in
# the 6.5-7.0% bucket from 2013 (3.9% in 2020), and the two swap over during
# January 2013. HNX shows the same handover from ~7% to ~10% in the same month.
BAND_REFORM = "2013-01-15"

BANDS_BEFORE = {"HOSE": 0.05, "HNX": 0.07, "UPCOM": 0.10}
BANDS_AFTER = {"HOSE": 0.07, "HNX": 0.10, "UPCOM": 0.15}

# The 2008 emergency, when the regulator repeatedly narrowed and then rewidened
# the limits to slow the crash. Visible in the data at weekly resolution: the
# 99th percentile of |return| on HOSE sits at 5.1% through 2008-03-24, drops to
# 2.02% the week of 03-31, runs at 2.9-3.1% into June, drifts up through 3.4-3.9%
# over the summer, and is back to 5.1% by 08-18.
#
# The exact schedule is NOT encoded, because reconstructing several successive
# changes from a percentile would be guesswork dressed as fact — and guessing
# TOO WIDE is the dangerous direction, since a limit-locked bar would then read
# as tradeable. The whole window is marked band-unknown instead, so those bars
# are excluded from trading rather than mis-sized.
BAND_UNKNOWN_PERIODS = [("2008-03-25", "2008-08-18")]


def band_known(date) -> bool:
    """False where the operative price limit cannot be stated with confidence."""
    d = str(date)[:10]
    return not any(lo <= d < hi for lo, hi in BAND_UNKNOWN_PERIODS)


# Kept as the current-era alias so callers that do not care about history read
# naturally; band() below is the one that knows about the reform.
BANDS = BANDS_AFTER
DEFAULT_BAND = 0.15  # unknown exchange: assume the widest, so we under-claim limits

# Cross-exchange threshold for "this move is too big to be legal anywhere".
# Used for data-quality flags only, never to classify a limit.
MAX_LEGAL_MOVE = 0.155

# The same idea for an INDEX, which is stricter: a broad index averages hundreds
# of names, each individually capped, so it cannot move like a single stock.
# Four VNINDEX prints break this (+120%, +84%, -55%, -43%) and are bad data.
MAX_INDEX_MOVE = 0.10

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


def band(exchange: str | None, date=None) -> float:
    """Price limit for an exchange on a given date.

    date=None means the current regime, which is what a caller asking about
    today wants. Anything reasoning over history must pass the bar's own date.
    """
    table = BANDS_AFTER
    if date is not None and str(date)[:10] < BAND_REFORM:
        table = BANDS_BEFORE
    return table.get(exchange or "", DEFAULT_BAND)


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
