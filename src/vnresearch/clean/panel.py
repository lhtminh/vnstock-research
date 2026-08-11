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
from vnresearch.clean import bands
from vnresearch.io import duck

# Statuses a backtest may transact on. Everything else either had no
# counterparty (limit-locked, halted, no trade) or is a bar we cannot vouch for.
TRADEABLE = "normal"


def _sql(adtv_window: int, adtv_min: float, require_full: bool) -> str:
    full_window = f"AND obs = {adtv_window}" if require_full else ""
    return f"""
WITH bars AS (
    -- A bar the classifier could not vouch for is barred from TRADING by
    -- `tradeable`, but that alone left its return feeding every window built on
    -- it. A suspect_move is a feed defect — some reach +682% — and one of those
    -- inside a 252-session volatility estimate corrupts the whole year.
    --
    -- OPEN, HIGH AND LOW GO TOO. If the return is not believed then neither is
    -- the range that produced it, and a suspect_ohlc bar is by definition one
    -- whose OHLC relationships are broken. These three feed only the range and
    -- volatility features — Parkinson, Garman-Klass, Rogers-Satchell, ATR,
    -- stochastic, Keltner, the channel — where a single bad high corrupts every
    -- window containing it, exactly as a bad return does. Nulling them makes
    -- those estimators skip the bar, which is the same treatment `ret` gets.
    --
    -- `open` is safe to drop despite being the label's ENTRY price: entry also
    -- requires `entry_tradeable`, and these bars are untradeable by definition,
    -- so no label that would have survived is lost.
    --
    -- CLOSE STAYS. It is needed to value a position and to compute turnover, and
    -- momentum features read it directly, so a defect landing on a window
    -- ENDPOINT can still reach them. That residual is accepted because the
    -- affected bars are 0.55% of the panel and the alternative breaks the LAG
    -- chain everywhere.
    SELECT * EXCLUDE (ret, open, high, low),
           CASE WHEN bar_status IN ('suspect_move', 'band_anomaly', 'suspect_ohlc')
                THEN NULL ELSE ret END AS ret,
           CASE WHEN bar_status IN ('suspect_move', 'band_anomaly', 'suspect_ohlc')
                THEN NULL ELSE open END AS open,
           CASE WHEN bar_status IN ('suspect_move', 'band_anomaly', 'suspect_ohlc')
                THEN NULL ELSE high END AS high,
           CASE WHEN bar_status IN ('suspect_move', 'band_anomaly', 'suspect_ohlc')
                THEN NULL ELSE low END AS low
    FROM read_parquet('{{bars}}')
),
-- The benchmark, with impossible prints removed.
--
-- A broad index cannot move 10% in a session: it is an average of hundreds of
-- names, each capped at its own band. Four VNINDEX bars break that — 2007-07-23
-- at -54.6%, 2007-07-24 at +120.5%, 2008-08-16 at +84.1% (a SATURDAY), and
-- 2008-08-19 at -43.2%. They are bad prints, not market moves.
--
-- They matter far beyond the four days. mkt_ret feeds beta_60, alpha_60,
-- idio_vol_60 and every excess_ret, so one defect corrupts a 60-session window
-- around it, and it enters the residual target directly. These dates sat
-- outside the sample until it was extended back to 2004, which is exactly the
-- kind of thing extending a sample drags in.
--
-- Nulled rather than repaired: the true level is unknown, and a guessed one
-- would propagate silently where a NULL simply removes the observation.
mkt AS (
    SELECT date, CASE WHEN abs(r) <= {bands.MAX_INDEX_MOVE} THEN r END AS mkt_ret
    FROM (
        SELECT date, close / NULLIF(LAG(close) OVER (ORDER BY date), 0) - 1 AS r
        FROM index_series WHERE index_code = 'VNINDEX'
    )
),
-- Dilution events: ESOP, rights issues and private placements, which sell NEW
-- shares and shrink every existing holder's stake. Stock dividends and bonus
-- issues are excluded upstream in v_share_events — they multiply the share
-- count and divide the price by the same factor, so the holder's stake is
-- unchanged and the adjusted price series already accounts for it.
--
-- Keyed on knowable_date, not ex_date: the market cannot react to an issue
-- before it is announced.
--
-- Summed as logs because several events can land on one date and the effect
-- compounds; the rolling window below turns that back into a multiplier.
ev AS (
    SELECT ticker, knowable_date AS date, SUM(LN(share_multiplier)) AS log_mult
    FROM v_share_events
    WHERE is_dilutive AND share_multiplier > 0
    GROUP BY ticker, knowable_date
),
-- Tet, derived rather than hard-coded. It moves on the lunar calendar between
-- late January and mid-February, so no fixed date or month works — but it is
-- always the market's longest closure of the year, which the calendar already
-- knows. Taking the largest January-to-March gap per year finds it without a
-- lunar library.
gaps AS (
    SELECT date,
           date_diff('day', LAG(date) OVER (ORDER BY date), date) AS gap,
           date_part('year', date) AS yr
    FROM (SELECT DISTINCT date FROM trading_calendar WHERE is_trading_day)
),
tet AS (
    SELECT yr, max_by(date, gap) AS resume_date
    FROM gaps
    WHERE gap IS NOT NULL AND date_part('month', date) BETWEEN 1 AND 3
    GROUP BY yr
),
-- Per-ticker calendar-month returns, the raw material for seasonality.
monthly AS (
    SELECT ticker,
           date_part('year', date)  AS yr,
           date_part('month', date) AS mo,
           AVG(ret) AS m_ret
    FROM bars
    WHERE ret IS NOT NULL
    GROUP BY 1, 2, 3
),
-- Per-ticker behaviour in the Tet window, one number per year.
--
-- This has to be per TICKER. Distance to Tet on its own is the same for every
-- stock on a date, so it cannot rank one above another — and multiplying
-- another feature by it does not help either, since a positive day-constant
-- leaves within-day ordering exactly as it was. Only "how does THIS stock
-- behave around Tet" varies across the cross-section.
tet_yearly AS (
    SELECT b.ticker,
           date_part('year', b.date) AS yr,
           AVG(b.ret) AS t_ret
    FROM bars b
    JOIN tet t ON t.yr = date_part('year', b.date)
    WHERE b.ret IS NOT NULL
      AND date_diff('day', t.resume_date, b.date) BETWEEN -14 AND 14
    GROUP BY 1, 2
),
tet_seas AS (
    SELECT ticker, yr,
           AVG(t_ret) OVER w AS seas_tet,
           COUNT(t_ret) OVER w AS tet_obs
    FROM tet_yearly
    WINDOW w AS (
        PARTITION BY ticker ORDER BY yr
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING   -- prior years only
    )
),
-- This ticker's own tendency in this calendar month, from PRIOR YEARS ONLY.
--
-- The frame is `5 PRECEDING AND 1 PRECEDING`: the current year is excluded, so
-- a February 2020 row is scored on February 2015-2019 and cannot see itself.
-- That exclusion is the whole point — including the current year would leak the
-- outcome into the feature.
--
-- Why this and not a month dummy: "it is February" is the same value for every
-- stock on the date, and a cross-sectional model cannot rank one stock above
-- another on a day-constant. "THIS stock usually does well in February" varies
-- across the universe, which is what makes it rankable.
seas AS (
    SELECT ticker, yr, mo,
           AVG(m_ret) OVER (
               PARTITION BY ticker, mo ORDER BY yr
               ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
           ) AS seas_month,
           COUNT(m_ret) OVER (
               PARTITION BY ticker, mo ORDER BY yr
               ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
           ) AS seas_obs
    FROM monthly
),
liq AS (
    SELECT
        b.*,
        COALESCE(ev.log_mult, 0.0) AS log_mult,
        -- No matched value in the source, so turnover is a close*volume proxy.
        -- It overstates limit-locked days and ignores put-through entirely.
        b.close * b.volume AS turnover,
        AVG(b.close * b.volume) OVER w AS adtv,
        COUNT(b.close * b.volume) OVER w AS obs
    FROM bars b
    LEFT JOIN ev ON ev.ticker = b.ticker AND ev.date = b.date
    WINDOW w AS (
        PARTITION BY b.ticker ORDER BY b.date
        ROWS BETWEEN {adtv_window - 1} PRECEDING AND CURRENT ROW
    )
),
dil AS (
    SELECT
        liq.*,
        -- Trailing-year dilution as a fraction: 0.05 means the share count grew
        -- 5% through issuance a holder did not participate in. Backward-only,
        -- like every other window here.
        EXP(SUM(log_mult) OVER (
            PARTITION BY ticker ORDER BY date
            ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
        )) - 1 AS dilution_252
    FROM liq
),
seasoned AS (
    SELECT
        dil.*,
        -- NULL until five prior same-month observations exist, rather than an
        -- average of two noisy years dressed up as a seasonal effect.
        CASE WHEN s.seas_obs >= 3 THEN s.seas_month END AS seas_month,
        -- This ticker's own Tet behaviour, from prior years. Applied only
        -- NEAR Tet: away from the holiday it describes nothing, and carrying
        -- it year-round would just add a stale constant per ticker.
        CASE WHEN ts.tet_obs >= 3
              AND ABS(date_diff('day', t.resume_date, dil.date)) <= 21
             THEN ts.seas_tet END AS seas_tet,
        -- Signed distance to the Tet reopening: negative before, positive
        -- after. Day-constant, so it is NOT registered as a feature — it is
        -- kept because seas_tet is built from it and because it makes the
        -- holiday visible when eyeballing the panel.
        date_diff('day', t.resume_date, dil.date) AS days_from_tet
    FROM dil
    LEFT JOIN seas s
           ON s.ticker = dil.ticker
          AND s.yr = date_part('year', dil.date)
          AND s.mo = date_part('month', dil.date)
    LEFT JOIN tet t ON t.yr = date_part('year', dil.date)
    LEFT JOIN tet_seas ts
           ON ts.ticker = dil.ticker
          AND ts.yr = date_part('year', dil.date)
)
SELECT
    s.ticker, s.date, s.open, s.high, s.low, s.close, s.volume,
    s.exchange, s.symbol_status, s.bar_status, s.prev_close, s.ret,
    s.turnover, s.adtv, s.obs AS adtv_obs, s.dilution_252,
    s.seas_month, s.seas_tet, s.days_from_tet, s.band_pct,
    (s.bar_status = '{TRADEABLE}') AS tradeable,
    (s.adtv >= {adtv_min} {full_window}) AS in_universe,
    mkt.mkt_ret
FROM seasoned s
LEFT JOIN mkt ON mkt.date = s.date
ORDER BY s.ticker, s.date
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
