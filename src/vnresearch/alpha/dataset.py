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


def open_joined(include_holdout: bool = False) -> duckdb.DuckDBPyConnection:
    """Connection with a `d` view: features joined to labels on (ticker, date).

    THE HOLDOUT IS EXCLUDED BY DEFAULT, and this was a real defect before it
    was: alpha analysis filtered on `in_universe` and nothing else, so every IC
    number was computed over data that included the frozen holdout. Features
    were then judged to "work" partly on the evidence they were supposed to be
    tested against later — which quietly spends the holdout without anyone
    deciding to.

    Pass include_holdout=True only for a deliberate final measurement.
    """
    feats = config.path("data/features/features.parquet")
    labels = config.path("data/clean/labels.parquet")
    for p in (feats, labels):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run the earlier stages first")

    cfg = config.load("model")["dataset"]
    where = ["f.in_universe"]
    if cfg.get("start_date"):
        where.append(f"f.date >= DATE '{cfg['start_date']}'")
    holdout = cfg.get("holdout_start")
    if holdout and not include_holdout:
        where.append(f"f.date < DATE '{holdout}'")

    con = duckdb.connect()
    con.execute(
        f"""CREATE VIEW d AS
            SELECT f.*, l.* EXCLUDE (ticker, date, in_universe, adtv, close)
            FROM read_parquet('{feats.as_posix()}') f
            JOIN read_parquet('{labels.as_posix()}') l
              ON l.ticker = f.ticker AND l.date = f.date
            WHERE {" AND ".join(where)}"""
    )
    return con
