"""Ratings: one Z-score per dimension, a composite, and a letter grade.

WHAT THIS IS FOR, AND WHAT IT IS NOT.

The LightGBM model in `model/` predicts. This EXPLAINS. A portfolio manager
looking at a name wants to know not just "is it ranked highly" but "on what" —
strong momentum and thin liquidity is a different position from the reverse,
even when both score the same overall. So every dimension keeps its own column
and the composite is a plain weighted sum of them, auditable by hand.

It is deliberately NOT a competitor to the model. Nothing here is fitted:
weights are equal by default, and each feature's direction is DECLARED in the
registry from what it measures, not fitted to the sample. `check_directions()`
reports where the data disagrees, which is a finding to discuss rather than a
sign to flip — a rating whose signs move between periods is one nobody can
explain, and explaining it is the whole job.

THREE THINGS THAT WOULD BE WRONG AND ARE NOT DONE HERE.

1. Z-scoring the RAW feature. A single stock with amihud 400x the median would
   own its dimension's mean. The registry already computes a within-day
   PERCENT_RANK of every feature; this z-scores THAT, so the input is bounded
   0..1 by construction and one name cannot dominate. `config/features.yaml`
   anticipated the problem: "Winsorization only matters for z-scores."
2. Ranking against the full sample. Invariant 4 — a stock is scored against the
   names trading that day, never against 2015.
3. Treating a missing feature as average. A NULL contributes nothing to its
   dimension's mean rather than a fabricated 0.5; if a whole dimension is
   missing, the dimension is NULL and the composite says so.
"""

from __future__ import annotations

import math
from pathlib import Path

import duckdb

from vnresearch import config
from vnresearch.features.registry import all_features

# Letter grades, best first, with the cumulative share of the day's universe
# each one covers. A is the top 10%, B the next 20%, and so on.
GRADES = [("A", 0.10), ("B", 0.30), ("C", 0.70), ("D", 0.90), ("E", 1.00)]


def _cfg() -> dict:
    return config.load("rating")


def dimensions() -> dict[str, list[tuple[str, int]]]:
    """{dimension: [(rank_column, direction), ...]} for every rated feature.

    Direction 0 means the feature is carried in the database but kept out of the
    rating — a volume spike has no sign until you know which way it broke, and
    averaging it in would add noise while looking like information.
    """
    out: dict[str, list[tuple[str, int]]] = {}
    for f in all_features().values():
        if f.direction:
            out.setdefault(f.category, []).append((f"{f.name}_rank", f.direction))
    return {k: sorted(v) for k, v in sorted(out.items())}


def _z(col: str, alias: str = "f") -> str:
    """Cross-sectional Z within the day. NULL stays NULL.

    Everything is qualified because the speculation table joins in beside the
    features and carries its own `date` — unqualified, the PARTITION BY is
    ambiguous and the query will not bind.

    NULLIF on the standard deviation, not a fabricated floor: on a day when
    every name shares a value the spread is genuinely zero and the Z-score is
    undefined, which is not the same as it being zero.
    """
    c, w = f"{alias}.{col}", f"OVER (PARTITION BY {alias}.date)"
    return f"({c} - AVG({c}) {w}) / NULLIF(STDDEV_SAMP({c}) {w}, 0)"


def _dimension_exprs(alias: str = "f") -> list[str]:
    """One averaged, sign-corrected Z per dimension.

    The average is over the features PRESENT on that row. Written as a sum of
    CASE-guarded terms over a count of the same, rather than an N-way average,
    because a name whose 252-session features have not warmed up yet should be
    scored on what it does have instead of dropping out of the rating entirely.
    """
    exprs = []
    for dim, feats in dimensions().items():
        terms = [f"{d} * COALESCE({_z(c, alias)}, 0)" for c, d in feats]
        present = [f"CASE WHEN {alias}.{c} IS NULL THEN 0 ELSE 1 END" for c, _ in feats]
        exprs.append(f"({' + '.join(terms)}) / NULLIF({' + '.join(present)}, 0) AS z_{dim}")
    return exprs


def build(verbose: bool = True) -> Path:
    """Write data/features/ratings.parquet."""
    cfg = _cfg()
    feats = config.path("data/features/features.parquet")
    if not feats.exists():
        raise FileNotFoundError(f"{feats} missing — run `vnr pipeline` first")

    dims = list(dimensions())
    weights = cfg["weights"]
    missing = [d for d in dims if d not in weights]
    if missing:
        raise ValueError(f"config/rating.yaml has no weight for: {', '.join(missing)}")

    # Weighted mean over the dimensions that exist on the row, so a warming-up
    # name is not silently scored as if its missing dimensions were zero.
    num = " + ".join(f"{weights[d]}::DOUBLE * COALESCE(z_{d}, 0)" for d in dims)
    den = " + ".join(f"CASE WHEN z_{d} IS NULL THEN 0 ELSE {weights[d]}::DOUBLE END" for d in dims)

    spec = config.path("data/clean/speculation.parquet")
    spec_col = cfg["speculation"]["label_column"]
    penalty = cfg["speculation"]["penalty_per_point"]
    if spec.exists():
        spec_join = f"""
        LEFT JOIN (SELECT ticker, date, spec_score, spec_score_adj, {spec_col} AS spec_label
                   FROM read_parquet('{spec.as_posix()}')) s
               ON s.ticker = f.ticker AND s.date = f.date"""
        spec_sel = "s.spec_score, s.spec_score_adj, s.spec_label"
        # The mentor's 0-3 points, scaled into Z units. NOT measured — a policy
        # lever. At the default 0.25 a name labelled Đầu cơ mạnh gives up 0.75
        # of a standard deviation, roughly the gap between a B and a C.
        #
        # Unqualified: by the time this is applied the column has been carried
        # through a CTE and the `s` alias no longer exists.
        spec_pen = f"COALESCE(spec_score, 0) * {penalty}::DOUBLE"
    else:
        spec_join = ""
        spec_sel = (
            "CAST(NULL AS DOUBLE) AS spec_score, CAST(NULL AS DOUBLE) AS spec_score_adj, "
            "CAST(NULL AS VARCHAR) AS spec_label"
        )
        spec_pen = "0.0"

    grade = (
        "CASE "
        + " ".join(f"WHEN pct <= {p} THEN '{g}'" for g, p in GRADES)
        + f" ELSE '{GRADES[-1][0]}' END"
    )

    out = config.path("data/features/ratings.parquet")
    con = duckdb.connect()
    try:
        con.execute(
            f"""COPY (
WITH z AS (
    SELECT f.ticker, f.date, f.close, f.adtv, f.exchange,
           {", ".join(_dimension_exprs())},
           {spec_sel}
    FROM read_parquet('{feats.as_posix()}') f{spec_join}
    WHERE f.in_universe
),
c AS (
    SELECT *,
           ({num}) / NULLIF({den}, 0) AS z_composite,
           count(*) OVER (PARTITION BY date) AS n_universe
    FROM z
),
r AS (
    SELECT *, z_composite - {spec_pen} AS rating_score FROM c
),
g AS (
    -- Ties broken by ticker: without it two names on the same score land in
    -- different grades on different runs. Invariant 7.
    SELECT *, PERCENT_RANK() OVER (PARTITION BY date ORDER BY rating_score DESC, ticker) AS pct
    FROM r
)
SELECT * EXCLUDE (pct), {grade} AS rating
FROM g
ORDER BY date, rating_score DESC
) TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)"""
        )
        if verbose:
            _summary(con, out.as_posix(), dims)
        return out
    finally:
        con.close()


def check_directions(horizon: int | None = None):
    """Where the declared direction disagrees with what the data did.

    Returns a frame, one row per rated feature, with the declared sign and the
    measured information coefficient. NOTHING IS FLIPPED — the point is to
    surface disagreements for a human to argue about, because a rating whose
    signs move with the sample is one nobody can explain, and explaining it is
    what the rating is for.

    IC here is the daily correlation between the feature's within-day rank and
    `fwd_rank`, which labels.parquet already computes — a percentile rank of the
    forward return inside the same day. Both sides are already ranks, so the
    plain correlation of the two IS the Spearman IC, without a sort per day.

    Dev period only. Reading the holdout to tune a sign would spend it.
    """
    import pandas as pd

    cfg = config.load("model")["dataset"]
    horizon = horizon or config.load("features")["label"]["horizon"]
    feats = config.path("data/features/features.parquet")
    labs = config.path("data/clean/labels.parquet")
    for p in (feats, labs):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run `vnr pipeline` first")

    rated = [(c, d, dim) for dim, fs in dimensions().items() for c, d in fs]
    where = [f"l.label_ok_{horizon}", f"l.fwd_rank_{horizon} IS NOT NULL", "f.in_universe"]
    if cfg.get("start_date"):
        where.append(f"f.date >= DATE '{cfg['start_date']}'")
    if h := cfg.get("holdout_start"):
        where.append(f"f.date < DATE '{h}'")

    # CORR returns NaN, not NULL, on a day where every name shares the feature's
    # value — trade_freq_21 is 1.0 for most of the universe most days. NaN then
    # propagates through the AVG and poisons the whole average, so the feature
    # reports as "disagrees" when in truth it was never measured. Same trap as
    # `label/speculation.finite()`, and invariant 3 once more.
    per_day = ", ".join(
        f"CASE WHEN isnan(CORR(f.{c}, l.fwd_rank_{horizon})) THEN NULL "
        f"ELSE CORR(f.{c}, l.fwd_rank_{horizon}) END AS {c}"
        for c, _, _ in rated
    )
    con = duckdb.connect()
    try:
        row = con.execute(
            f"""WITH d AS (
                    SELECT f.date, {per_day}
                    FROM read_parquet('{feats.as_posix()}') f
                    JOIN read_parquet('{labs.as_posix()}') l
                      ON l.ticker = f.ticker AND l.date = f.date
                    WHERE {" AND ".join(where)}
                    GROUP BY f.date HAVING count(*) >= 30)
                SELECT {", ".join(f"AVG({c})" for c, _, _ in rated)} FROM d"""
        ).fetchone()
    finally:
        con.close()

    out = pd.DataFrame(
        {
            "feature": [c.removesuffix("_rank") for c, _, _ in rated],
            "dimension": [dim for _, _, dim in rated],
            "declared": [d for _, d, _ in rated],
            "ic": [None if v is None or math.isnan(v) else round(float(v), 4) for v in row],
        }
    )
    out["agrees"] = [
        None if ic is None else (ic > 0) == (d > 0) for ic, d in zip(out["ic"], out["declared"])
    ]
    return out.sort_values(["dimension", "feature"]).reset_index(drop=True)


def _summary(con, path: str, dims: list[str]) -> None:
    n, first, last = con.execute(
        f"SELECT count(*), min(date), max(date) FROM read_parquet('{path}')"
    ).fetchone()
    print(f"  {n:,} rows  {first} .. {last}")
    print(f"  {len(dims)} dimensions: {', '.join(dims)}")
    rows = con.execute(
        f"""SELECT rating, count(*) n, round(avg(rating_score), 3) avg_score
            FROM read_parquet('{path}') WHERE rating IS NOT NULL
            GROUP BY 1 ORDER BY 1"""
    ).fetchall()
    print("\n  grade      rows      mean score")
    for g, cnt, avg in rows:
        print(f"    {g}    {cnt:>9,}    {avg:>+8.3f}")
