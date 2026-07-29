"""Forward-return labels that a strategy could actually have earned.

A signal computed on the close of day t cannot transact at that close. So the
label is measured from the OPEN of the next session to the open N sessions
later, and the entry bar must be tradeable — on a limit-up day there is no
offer side to buy from.

Two decisions worth knowing about:

ENTRY is strict. If the next session is limit-locked, halted or had no trade,
the sample is dropped. Buying there is impossible, so a label that assumes you
did is fiction. Entry is shared across horizons — you buy once.

EXIT is not strict, on purpose. Dropping samples whose exit day was limit-up
would remove exactly the trades that worked, truncating the right tail and
teaching the model to avoid momentum. The exit price is used as-is and
`exit_tradeable_N` is kept so the effect can be measured rather than silently
baked in.

Several horizons are labelled at once so IC decay can be measured without
rebuilding the file.
"""

from __future__ import annotations

from pathlib import Path

from vnresearch import config
from vnresearch.io import duck

# A ticker can be suspended for months. LEAD() would then reach across the gap
# and price an "exit" a quarter later, so bound the calendar distance.
MAX_ENTRY_GAP_DAYS = 12  # covers the Tet holiday
MAX_EXIT_GAP_PER_SESSION = 3


def _horizon_cols(h: int, price_col: str) -> tuple[str, str, str]:
    """(lead columns, validity expression, output columns) for one horizon."""
    leads = f"""
        LEAD({price_col}, {1 + h}) OVER w AS exit_px_{h},
        LEAD(date, {1 + h})        OVER w AS exit_date_{h},
        LEAD(tradeable, {1 + h})   OVER w AS exit_tradeable_{h}"""

    gap = MAX_ENTRY_GAP_DAYS + MAX_EXIT_GAP_PER_SESSION * h
    ok = f"""
        (entry_ok
         AND exit_px_{h} IS NOT NULL AND exit_px_{h} > 0
         AND date_diff('day', entry_date, exit_date_{h}) <= {gap}) AS label_ok_{h}"""

    out = f"""
        exit_px_{h}, exit_date_{h}, exit_tradeable_{h}, label_ok_{h},
        CASE WHEN label_ok_{h} THEN exit_px_{h} / entry_px - 1 END AS fwd_ret_{h},
        CASE WHEN label_ok_{h} AND in_universe THEN
            PERCENT_RANK() OVER (
                PARTITION BY date, (label_ok_{h} AND in_universe)
                ORDER BY CASE WHEN label_ok_{h} THEN exit_px_{h} / entry_px - 1 END
            )
        END AS fwd_rank_{h}"""
    return leads, ok, out


def _sql(horizons: list[int], primary: int, price_col: str) -> str:
    leads, oks, outs = zip(*(_horizon_cols(h, price_col) for h in horizons))
    return f"""
WITH fwd AS (
    SELECT
        ticker, date, in_universe, tradeable, bar_status, adtv, close,
        LEAD({price_col}, 1) OVER w AS entry_px,
        LEAD(date, 1)        OVER w AS entry_date,
        LEAD(tradeable, 1)   OVER w AS entry_tradeable,
        {",".join(leads)}
    FROM read_parquet('{{panel}}')
    WINDOW w AS (PARTITION BY ticker ORDER BY date)
),
entry AS (
    SELECT *,
        (entry_px IS NOT NULL AND entry_px > 0
         AND entry_tradeable
         AND date_diff('day', date, entry_date) <= {MAX_ENTRY_GAP_DAYS}) AS entry_ok
    FROM fwd
),
valid AS (
    SELECT *, {",".join(oks)} FROM entry
)
SELECT
    ticker, date, entry_date, entry_px, entry_tradeable, entry_ok,
    in_universe, adtv, close,
    {",".join(outs)},
    -- Primary horizon, aliased so the model and backtest need not know which.
    fwd_ret_{primary}  AS fwd_ret,
    fwd_rank_{primary} AS fwd_rank,
    label_ok_{primary} AS label_ok
FROM valid
ORDER BY ticker, date
"""


def build(verbose: bool = True) -> Path:
    """Write data/clean/labels.parquet."""
    cfg = config.load("features")["label"]
    primary = cfg["horizon"]
    horizons = sorted(set(cfg.get("horizons", [primary])) | {primary})
    price_col = cfg["entry_price"]

    clean_dir = config.path("data/clean")
    panel_path = clean_dir / "panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"{panel_path} missing — run `vnr panel` first")

    target = clean_dir / "labels.parquet"
    sql = _sql(horizons, primary, price_col).replace("{panel}", panel_path.as_posix())

    con = duck.open_mirror()
    try:
        con.execute(f"COPY ({sql}) TO '{target.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
        if verbose:
            print(f"  entry               next {price_col}, must be tradeable")
            print(f"  horizons            {horizons} (primary {primary})")
            for h in horizons:
                uni, ok, mean, sd = con.execute(
                    f"""SELECT count(*) FILTER (WHERE in_universe),
                               count(*) FILTER (WHERE in_universe AND label_ok_{h}),
                               round(100*avg(fwd_ret_{h}) FILTER (WHERE in_universe), 3),
                               round(100*stddev(fwd_ret_{h}) FILTER (WHERE in_universe), 3)
                        FROM read_parquet('{target.as_posix()}')"""
                ).fetchone()
                print(
                    f"    h={h:<3} labelled {ok:>8,}/{uni:,} ({100 * ok / uni:4.1f}%)"
                    f"  mean {mean:>6}%  sd {sd:>6}%"
                )
    finally:
        con.close()
    return target
