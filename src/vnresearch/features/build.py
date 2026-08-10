"""Compute every registered feature over the panel and write features.parquet.

Two passes:
  1. raw    — the registry expressions, each NULLed until its lookback is met
  2. xsec   — percentile rank of each raw feature within the day's universe

The warm-up NULLing matters. A window aggregate happily returns a value from 3
observations when the feature was defined over 252, and that value looks like
every other number in the column. Blanking it keeps "not enough history" and "a
real reading" distinguishable.
"""

from __future__ import annotations

from pathlib import Path

from vnresearch import config
from vnresearch.features import xsec
from vnresearch.features.registry import all_features
from vnresearch.io import duck

# Carried through from the panel so downstream never needs to re-join.
_KEYS = [
    "ticker",
    "date",
    "close",
    "ret",
    "adtv",
    "exchange",
    "bar_status",
    "tradeable",
    "in_universe",
]


def _raw_sql(panel_path: str, join_peers: bool = True) -> tuple[str, list[str]]:
    feats = all_features()
    cols = []
    for f in sorted(feats.values(), key=lambda x: (x.category, x.name)):
        # session_idx counts this ticker's own bars, so a feature only appears
        # once the ticker has the history its definition assumes.
        cols.append(f"CASE WHEN session_idx >= {f.lookback} THEN ({f.sql}) END AS {f.name}")

    # Peer features are computed in pandas (correlation matrices are awkward in
    # SQL) and joined here rather than in the panel, because peers.py reads the
    # panel and joining there would be circular.
    #
    # join_peers=False is for callers whose panel ALREADY carries the peer
    # columns — the tests build a synthetic one that way. Joining as well would
    # give two columns of the same name, and which one the feature SQL then
    # resolves to is not something to leave to chance.
    peers_path = config.path("data/features/peers.parquet")
    if join_peers and peers_path.exists():
        peer_join = f"""
    LEFT JOIN read_parquet('{peers_path.as_posix()}') pr
           ON pr.ticker = b.ticker AND pr.date = b.date"""
        peer_cols = "pr.peer_ret_1, pr.peer_ret_5, pr.peer_ret_21, pr.peer_corr, pr.peer_n"
    else:
        peer_join = ""
        peer_cols = """CAST(NULL AS DOUBLE) AS peer_ret_1,
                       CAST(NULL AS DOUBLE) AS peer_ret_5,
                       CAST(NULL AS DOUBLE) AS peer_ret_21,
                       CAST(NULL AS DOUBLE) AS peer_corr,
                       CAST(NULL AS BIGINT)  AS peer_n"""

    sql = f"""
WITH j AS (
    SELECT b.*, {peer_cols}
    FROM read_parquet('{panel_path}') b{peer_join}
),
p AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date) AS session_idx
    FROM j
)
SELECT {", ".join(_KEYS)}, session_idx,
       {", ".join(cols)}
FROM p
"""
    return sql, sorted(feats.keys())


def _xsec_sql(raw_path: str, names: list[str]) -> str:
    ranks = [f"{xsec.rank_expr(n)} AS {n}_rank" for n in names]
    # Only universe rows survive. Everything downstream trains, ranks and
    # trades inside the universe, and the panel still holds prices for any
    # position that later drops out of it. Keeping the rest would quintuple
    # the file for rows nothing reads.
    return f"""
SELECT * EXCLUDE (session_idx)
FROM (
    SELECT *, {", ".join(ranks)}
    FROM read_parquet('{raw_path}')
)
WHERE in_universe
"""


def build(verbose: bool = True) -> Path:
    """Write data/features/features.parquet."""
    panel = config.path("data/clean/panel.parquet")
    if not panel.exists():
        raise FileNotFoundError(f"{panel} missing — run `vnr panel` first")

    out = config.path("data/features")
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "features_raw.parquet"
    target = out / "features.parquet"

    raw_sql, names = _raw_sql(panel.as_posix())
    con = duck.open_mirror()
    try:
        con.execute(
            f"COPY ({raw_sql}) TO '{raw_path.as_posix()}' (FORMAT parquet, COMPRESSION zstd)"
        )
        con.execute(
            f"COPY ({_xsec_sql(raw_path.as_posix(), names)}) "
            f"TO '{target.as_posix()}' (FORMAT parquet, COMPRESSION zstd)"
        )
        if verbose:
            print(f"  {len(names)} features x 2 (raw + rank)")
            _coverage(con, target.as_posix(), names)
    finally:
        con.close()
    raw_path.unlink(missing_ok=True)
    return target


def _coverage(con, path: str, names: list[str]) -> None:
    """Report how much of the universe each feature actually covers.

    An all-NULL feature is a bug; a partly-NULL one is usually a real data
    boundary (market features start in 2019) and should be recognised as such.
    """
    parts = [
        f"round(100.0 * count({n}) FILTER (WHERE in_universe) "
        f"/ NULLIF(count(*) FILTER (WHERE in_universe), 0), 1) AS {n}"
        for n in names
    ]
    row = con.execute(f"SELECT {', '.join(parts)} FROM read_parquet('{path}')").fetchone()
    thin = [(n, v) for n, v in zip(names, row) if v is None or v < 90]
    print(f"  coverage >=90% on {len(names) - len(thin)}/{len(names)} features")
    for n, v in sorted(thin, key=lambda x: (x[1] is None, x[1])):
        print(f"    thin: {n:<22} {v}%")
