"""Rebuild bar_status for every bar, since the source stores it as NULL.

Runs as one DuckDB query over the Parquet mirror — 4.8M bars in a few seconds.
The SQL is generated from the constants in bands.py so there is one source of
truth for the limits.

Status precedence (first match wins):

  no_trade      volume is 0 or missing
  suspect_ohlc  high/low contradict open/close
  unknown       no usable reference price
  limit_up/down closed at the price limit
  normal        everything else — the only status a backtest may fill on

no_trade is checked before suspect_ohlc on purpose: a session with no trades
reports untraded reference prices as OHLC, so incoherence there is expected
rather than a defect.
"""

from __future__ import annotations

from pathlib import Path

from vnresearch import config
from vnresearch.clean import bands
from vnresearch.io import duck

# SQL mirrors bands.tick_size(). Keep the two in step — tests assert they agree.
_TICK_SQL = """
    CASE
        WHEN s.exchange = 'HOSE' AND lagged.prev_close <  10000 THEN 10
        WHEN s.exchange = 'HOSE' AND lagged.prev_close <  50000 THEN 50
        ELSE 100
    END
"""


# Date-dependent, because the bands widened on 2013-01-15 (see bands.py). Using
# today's numbers on a 2010 bar reclassifies every real limit-up as an ordinary
# session and lets the backtest fill where there was no offer side.
#
# Cast to DOUBLE: bare decimal literals bind as DECIMAL in DuckDB, which then
# mixes with the DOUBLE returns in every comparison below.
def _band_sql() -> str:
    def table(d: dict[str, float]) -> str:
        return f"""CASE s.exchange
            WHEN 'HOSE'  THEN {d["HOSE"]}::DOUBLE
            WHEN 'HNX'   THEN {d["HNX"]}::DOUBLE
            WHEN 'UPCOM' THEN {d["UPCOM"]}::DOUBLE
            ELSE {bands.DEFAULT_BAND}::DOUBLE
        END"""

    return f"""
    CASE WHEN lagged.date < DATE '{bands.BAND_REFORM}'
         THEN {table(bands.BANDS_BEFORE)}
         ELSE {table(bands.BANDS_AFTER)}
    END
"""


_BAND_SQL = _band_sql()


def _sql() -> str:
    return f"""
WITH lagged AS (
    SELECT
        p.ticker, p.date, p.open, p.high, p.low, p.close, p.volume,
        p.adjustment_epoch,
        LAG(p.close) OVER w AS prev_close,
        LAG(p.date)  OVER w AS prev_date
    FROM daily_prices p
    WINDOW w AS (PARTITION BY p.ticker ORDER BY p.date)
),
b AS (
    SELECT
        lagged.*,
        s.exchange,
        s.status AS symbol_status,
        {_BAND_SQL} AS band,
        {_TICK_SQL} AS tick,
        lagged.close / NULLIF(lagged.prev_close, 0) - 1 AS ret,
        date_diff('day', lagged.prev_date, lagged.date) AS ref_gap_days
    FROM lagged
    LEFT JOIN symbols s ON s.ticker = lagged.ticker
),
t AS (
    SELECT
        b.*,
        LEAST(GREATEST(b.tick / NULLIF(b.prev_close, 0), {bands.MIN_TOLERANCE}),
              {bands.MAX_TOLERANCE}) AS tol,
        -- Reference price we can actually reason about.
        (b.prev_close IS NOT NULL
         AND b.prev_close >= {bands.MIN_PRICE_FOR_BAND}
         AND b.ref_gap_days <= {bands.MAX_REF_GAP_DAYS}) AS ref_ok
    FROM b
)
SELECT
    ticker, date, open, high, low, close, volume, adjustment_epoch,
    exchange, symbol_status, prev_close, ret, ref_gap_days,
    CASE
        WHEN volume IS NULL OR volume = 0 THEN 'no_trade'
        WHEN high < GREATEST(open, close) - 0.01
          OR low  > LEAST(open, close) + 0.01 THEN 'suspect_ohlc'
        WHEN NOT ref_ok THEN 'unknown'
        -- Bigger than any exchange allows, so not a price move at all: an
        -- adjustment defect. Must be caught BEFORE the limit tests, which
        -- otherwise read a +682% jump as a limit-up.
        WHEN abs(ret) > {bands.MAX_LEGAL_MOVE} THEN 'suspect_move'
        WHEN ret BETWEEN  band - tol AND  band + tol THEN 'limit_up'
        WHEN ret BETWEEN -band - tol AND -band + tol THEN 'limit_down'
        -- Legal somewhere, but not on the exchange we think this ticker is on.
        -- symbols.exchange is the CURRENT venue, so an old bar from a previous
        -- listing lands here. Not tradeable: we cannot tell what the band was.
        WHEN abs(ret) > band + tol THEN 'band_anomaly'
        ELSE 'normal'
    END AS bar_status
FROM t
ORDER BY ticker, date
"""


def build(out_dir: str | Path | None = None, verbose: bool = True) -> Path:
    """Write data/clean/bars.parquet with bar_status attached to every bar."""
    out = Path(out_dir) if out_dir else config.path("data/clean")
    out.mkdir(parents=True, exist_ok=True)
    target = out / "bars.parquet"

    con = duck.open_mirror()
    try:
        con.execute(f"COPY ({_sql()}) TO '{target.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
        if verbose:
            rows = con.execute(
                f"""SELECT bar_status, count(*) n,
                           round(100.0 * count(*) / sum(count(*)) OVER (), 2) pct
                    FROM read_parquet('{target.as_posix()}')
                    GROUP BY 1 ORDER BY 2 DESC"""
            ).fetchall()
            for status, n, pct in rows:
                print(f"  {status:<14} {n:>9,}  {pct:5.2f}%")
    finally:
        con.close()
    return target
