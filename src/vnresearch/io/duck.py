"""DuckDB access to the vnstock-service database, and the local Parquet mirror.

Feature engineering makes many passes over the same 4.8M bars, so we copy
Postgres to Parquet once and work locally. The mirror is also a snapshot: a
research run is reproducible, and it does not break when the 15:30 ingestion
task writes mid-run.
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb

from vnresearch import config
from vnresearch.io.manifest import Manifest, write_manifest

# Postgres NUMERIC becomes DECIMAL in DuckDB, which lands in pandas as Decimal
# objects and quietly breaks arithmetic. Cast to DOUBLE on the way out.
_SELECT: dict[str, str] = {
    "daily_prices": """
        SELECT ticker, date,
               open::DOUBLE  AS open,
               high::DOUBLE  AS high,
               low::DOUBLE   AS low,
               close::DOUBLE AS close,
               volume, is_adjusted, adjustment_epoch, source, ingested_at
        FROM pg.daily_prices
    """,
    "index_series": """
        SELECT index_code, date,
               open::DOUBLE AS open, high::DOUBLE AS high,
               low::DOUBLE  AS low,  close::DOUBLE AS close,
               volume, value
        FROM pg.index_series
    """,
    "market_snapshot": """
        SELECT ticker, date, exchange,
               ref_price::DOUBLE     AS ref_price,
               ceiling_price::DOUBLE AS ceiling_price,
               floor_price::DOUBLE   AS floor_price,
               match_price::DOUBLE   AS match_price,
               volume, value,
               foreign_buy_volume, foreign_sell_volume,
               foreign_buy_value, foreign_sell_value,
               foreign_room, total_room, listed_share,
               trading_status, bar_status, captured_at
        FROM pg.market_snapshot
    """,
    "company_snapshot": """
        SELECT ticker, date,
               current_price::DOUBLE   AS current_price,
               market_cap, issue_share,
               foreign_pct::DOUBLE     AS foreign_pct,
               max_foreign_pct::DOUBLE AS max_foreign_pct,
               state_pct::DOUBLE       AS state_pct,
               avg_match_value_1m, avg_match_volume_1m,
               high_52w::DOUBLE AS high_52w, low_52w::DOUBLE AS low_52w
        FROM pg.company_snapshot
    """,
    "corporate_actions": """
        SELECT ticker, action_type, sub_type, ex_date, record_date, pay_date,
               announced_date,
               cash_amount::DOUBLE AS cash_amount,
               ratio_from::DOUBLE  AS ratio_from,
               ratio_to::DOUBLE    AS ratio_to,
               event_id, raw_title
        FROM pg.corporate_actions
    """,
}


def connect(attach_pg: bool = False, read_only_db: str | None = None) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection, optionally attaching the Postgres source."""
    con = duckdb.connect(read_only_db or ":memory:")
    if attach_pg:
        con.execute("INSTALL postgres; LOAD postgres;")
        con.execute(f"ATTACH '{config.dsn()}' AS pg (TYPE postgres, READ_ONLY);")
    return con


def mirror(tables: list[str] | None = None, verbose: bool = True) -> Manifest:
    """Copy Postgres tables to Parquet and write a manifest describing the snapshot."""
    cfg = config.load("data")
    tables = tables or cfg["tables"]
    out = config.path(cfg["mirror_dir"])
    out.mkdir(parents=True, exist_ok=True)

    con = connect(attach_pg=True)
    counts: dict[str, int] = {}

    for t in tables:
        select = _SELECT.get(t, f"SELECT * FROM pg.{t}")
        target = out / f"{t}.parquet"
        t0 = time.time()
        con.execute(f"COPY ({select}) TO '{target.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
        n = con.execute(f"SELECT count(*) FROM read_parquet('{target.as_posix()}')").fetchone()[0]
        counts[t] = n
        if verbose:
            mb = target.stat().st_size / 1e6
            print(f"  {t:<20} {n:>9,} rows  {mb:6.1f} MB  {time.time() - t0:5.1f}s")

    epochs = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT adjustment_epoch FROM pg.daily_prices ORDER BY 1"
        ).fetchall()
    ]
    max_ingest = con.execute("SELECT max(ingested_at) FROM pg.daily_prices").fetchone()[0]
    # Per-ticker seams. Several epochs across the TABLE is normal — repairing a
    # restatement bumps one ticker at a time — but a single ticker holding two
    # is a fabricated overnight gap in its own series. Mirrors v_basis_seams.
    seams = [
        r[0]
        for r in con.execute(
            """SELECT ticker FROM pg.daily_prices
               WHERE adjustment_epoch IS NOT NULL
               GROUP BY ticker HAVING count(DISTINCT adjustment_epoch) > 1
               ORDER BY ticker"""
        ).fetchall()
    ]
    con.close()

    m = Manifest(
        row_counts=counts,
        adjustment_epochs=[e for e in epochs if e is not None],
        max_ingested_at=str(max_ingest) if max_ingest else None,
        seam_tickers=seams,
    )
    write_manifest(out, m)
    return m


def open_mirror(mirror_dir: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """Connection with every mirrored Parquet file registered as a view of the same name."""
    cfg = config.load("data")
    out = Path(mirror_dir) if mirror_dir else config.path(cfg["mirror_dir"])
    if not out.exists():
        raise FileNotFoundError(f"no mirror at {out} — run `vnr mirror` first")

    con = duckdb.connect()
    for f in sorted(out.glob("*.parquet")):
        con.execute(
            f"CREATE OR REPLACE VIEW {f.stem} AS SELECT * FROM read_parquet('{f.as_posix()}')"
        )
    return con


def read(table: str, mirror_dir: str | Path | None = None):
    """Read one mirrored table into a pandas DataFrame."""
    con = open_mirror(mirror_dir)
    try:
        return con.execute(f"SELECT * FROM {table}").df()
    finally:
        con.close()
