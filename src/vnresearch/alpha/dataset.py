"""The joined features + labels table that alpha analysis and training share."""

from __future__ import annotations

import duckdb

from vnresearch import config
from vnresearch.features.registry import all_features

MIN_NAMES_PER_DAY = 30  # a cross-sectional statistic from 5 names is noise


def feature_names(ranked: bool = True) -> list[str]:
    """Registered feature columns, ranked versions by default."""
    names = sorted(all_features().keys())
    return [f"{n}_rank" for n in names] if ranked else names


def open_joined() -> duckdb.DuckDBPyConnection:
    """Connection with a `d` view: features joined to labels on (ticker, date)."""
    feats = config.path("data/features/features.parquet")
    labels = config.path("data/clean/labels.parquet")
    for p in (feats, labels):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run the earlier stages first")

    con = duckdb.connect()
    con.execute(
        f"""CREATE VIEW d AS
            SELECT f.*, l.* EXCLUDE (ticker, date, in_universe, adtv, close)
            FROM read_parquet('{feats.as_posix()}') f
            JOIN read_parquet('{labels.as_posix()}') l
              ON l.ticker = f.ticker AND l.date = f.date
            WHERE f.in_universe"""
    )
    return con
