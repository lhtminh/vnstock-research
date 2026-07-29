"""The base panel: one row per (ticker, date) with everything that is not a
model feature — prices, tradeability, liquidity, universe membership, and the
market return.

Labels and features both read this, so they can never disagree about which
bars were tradeable or who was in the universe on a given day.

Point-in-time rule for everything here: windows are `ROWS BETWEEN n PRECEDING
AND CURRENT ROW`. Nothing may look forward. The one deliberate exception is the
label itself, which lives in label/forward.py and is clearly named.
"""

from __future__ import annotations

from pathlib import Path

from vnresearch import config
from vnresearch.io import duck

# Statuses a backtest may transact on. Everything else either had no
# counterparty (limit-locked, halted, no trade) or is a bar we cannot vouch for.
TRADEABLE = "normal"


def _sql(adtv_window: int, adtv_min: float, require_full: bool) -> str:
    full_window = f"AND obs = {adtv_window}" if require_full else ""
    return f"""
WITH bars AS (
    SELECT * FROM read_parquet('{{bars}}')
),
mkt AS (
    SELECT date,
           close / NULLIF(LAG(close) OVER (ORDER BY date), 0) - 1 AS mkt_ret
    FROM index_series WHERE index_code = 'VNINDEX'
),
liq AS (
    SELECT
        b.*,
        -- No matched value in the source, so turnover is a close*volume proxy.
        -- It overstates limit-locked days and ignores put-through entirely.
        b.close * b.volume AS turnover,
        AVG(b.close * b.volume) OVER w AS adtv,
        COUNT(b.close * b.volume) OVER w AS obs
    FROM bars b
    WINDOW w AS (
        PARTITION BY b.ticker ORDER BY b.date
        ROWS BETWEEN {adtv_window - 1} PRECEDING AND CURRENT ROW
    )
)
SELECT
    liq.ticker, liq.date, liq.open, liq.high, liq.low, liq.close, liq.volume,
    liq.exchange, liq.symbol_status, liq.bar_status, liq.prev_close, liq.ret,
    liq.turnover, liq.adtv, liq.obs AS adtv_obs,
    (liq.bar_status = '{TRADEABLE}') AS tradeable,
    (liq.adtv >= {adtv_min} {full_window}) AS in_universe,
    mkt.mkt_ret
FROM liq
LEFT JOIN mkt ON mkt.date = liq.date
ORDER BY liq.ticker, liq.date
"""


def build(verbose: bool = True) -> Path:
    """Write data/clean/panel.parquet."""
    cfg = config.load("features")["universe"]
    clean_dir = config.path("data/clean")
    bars_path = clean_dir / "bars.parquet"
    if not bars_path.exists():
        raise FileNotFoundError(f"{bars_path} missing — run `vnr clean` first")

    target = clean_dir / "panel.parquet"
    sql = _sql(cfg["adtv_window"], float(cfg["adtv_min_vnd"]), cfg["require_full_window"])
    sql = sql.replace("{bars}", bars_path.as_posix())

    con = duck.open_mirror()
    try:
        con.execute(f"COPY ({sql}) TO '{target.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
        if verbose:
            n, uni, tradeable = con.execute(
                f"""SELECT count(*),
                           count(*) FILTER (WHERE in_universe),
                           count(*) FILTER (WHERE in_universe AND tradeable)
                    FROM read_parquet('{target.as_posix()}')"""
            ).fetchone()
            print(f"  rows                {n:>9,}")
            print(f"  in universe         {uni:>9,}  ({100 * uni / n:.1f}%)")
            print(
                f"  ...and tradeable    {tradeable:>9,}  ({100 * tradeable / uni:.1f}% of universe)"
            )
            names = con.execute(
                f"""SELECT count(DISTINCT ticker) FROM read_parquet('{target.as_posix()}')
                    WHERE in_universe AND date >= '2024-01-01'"""
            ).fetchone()[0]
            print(f"  distinct names 2024+{names:>9,}")
    finally:
        con.close()
    return target
