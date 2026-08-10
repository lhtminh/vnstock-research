"""Publish the labelled sample into Postgres, in a `research` schema.

Parquet stays the working format — it is faster for the many passes feature
engineering makes, and it is a snapshot. This copies the two artefacts that
carry the labelled sample into the database so anything that is NOT this repo
can reach it with plain SQL.

WHAT IS NOT TOUCHED. Everything lands in `research`. `public` belongs to
vnstock-service, which owns its own migrations and applies them from Go; this
never writes there, and a separate schema rather than a table prefix is what
makes that mechanical instead of a promise. The service's `EnsureSchema` runs
unqualified DDL, so it can only ever touch `public`.

WHY THE FEATURE COLUMNS ARE REAL, NOT DOUBLE PRECISION. `dataset.load()` casts
every feature to float32 before the model sees one, so a stored double carries
bits nothing consumes. 505 MB instead of 918 MB, and the value the model reads
is identical — both paths round the same double to the same float32. Prices,
returns and the labels themselves stay double.

THE VIEWS ARE THE POINT. `research.features` and `research.labels` are two
tables that still have to be joined and filtered correctly to be a training
set. `research.training_sample` is that join, generated from the same config and
the same target expression `dataset.load()` uses, so what you get from SQL is
the matrix the model actually trains on rather than something shaped like it.
`test_publish.py` asserts the two agree row for row.

The holdout is a SEPARATE view for the reason given in invariant 10: it is the
only measurement development never saw, and querying it should take a deliberate
act rather than a forgotten date filter.
"""

from __future__ import annotations

import time
from typing import Any

from vnresearch import config
from vnresearch.io.manifest import load_manifest

SCHEMA = "research"


def _connect():
    """DuckDB with the target Postgres attached WRITABLE.

    `duck.connect(attach_pg=True)` attaches READ_ONLY, which is right for every
    other caller here and wrong for this one.
    """
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL postgres; LOAD postgres;")
    con.execute(f"ATTACH '{config.dsn()}' AS pg (TYPE postgres);")
    return con


def _exec(con, sql: str) -> None:
    """Run DDL in Postgres itself rather than through DuckDB's table layer.

    Views, indexes and comments have no DuckDB equivalent to forward, so they go
    over as native statements. The doubling escapes the statement for the DuckDB
    string literal it travels in; anything quoted INSIDE the statement must
    already be valid Postgres before it gets here — see `_literal`.
    """
    con.execute(f"CALL postgres_execute('pg', '{sql.replace(chr(39), chr(39) * 2)}')")


def _literal(text: str) -> str:
    """A Postgres string literal. Comment text contains apostrophes."""
    return "'" + text.replace("'", "''") + "'"


def _feature_columns(con, path: str) -> list[str]:
    """The SELECT list for research.features, narrowing features to REAL.

    Derived from the registry rather than a hardcoded list of identity columns,
    so a feature added tomorrow is narrowed without anyone remembering to.
    """
    from vnresearch.features.registry import all_features

    narrow = {n for name in all_features() for n in (name, f"{name}_rank")}
    cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
    out = []
    for name, dtype, *_ in cols:
        out.append(f"{name}::REAL AS {name}" if name in narrow and dtype == "DOUBLE" else name)
    return out


def _sample_view_sql(name: str, holdout_only: bool) -> str:
    """The training matrix as a view — a transcription of `dataset.load()`.

    Same feature list, same filters, same target expression, same within-day
    PERCENT_RANK. Imported rather than restated: `stress.py` kept its own copy of
    the target and quietly scored the old one after it changed.
    """
    from vnresearch.alpha.dataset import feature_names
    from vnresearch.model.dataset import _target_sql, target_filters

    cfg = config.load("model")["dataset"]
    h = config.load("features")["label"]["horizon"]
    target = cfg.get("target", "residual")
    feats = feature_names(ranked=cfg["use_ranked_features"])

    where = [f"label_ok_{h}", f"fwd_ret_{h} IS NOT NULL", *target_filters(h, target)]
    if cfg.get("start_date"):
        where.append(f"date >= DATE '{cfg['start_date']}'")
    if holdout := cfg.get("holdout_start"):
        where.append(f"date >= DATE '{holdout}'" if holdout_only else f"date < DATE '{holdout}'")

    return f"""
CREATE VIEW {SCHEMA}.{name} AS
WITH j AS (
    SELECT f.*, l.fwd_ret_{h}, l.bench_ret_{h}, l.label_ok_{h}
    FROM {SCHEMA}.features f
    JOIN {SCHEMA}.labels l ON l.ticker = f.ticker AND l.date = f.date
),
base AS (
    SELECT ticker, date, {", ".join(feats)},
           fwd_ret_{h} AS fwd_ret,
           bench_ret_{h} AS bench_ret,
           ({_target_sql(h, target)}) AS target
    FROM j
    WHERE {" AND ".join(where)}
)
SELECT *, PERCENT_RANK() OVER (PARTITION BY date ORDER BY target, ticker) AS y
FROM base"""


def _write_catalog(con) -> int:
    """One row per feature: which dimension it belongs to, and its definition.

    Without this the published matrix is 127 unlabelled columns. The point of the
    feature work was to see the market from several dimensions at once, and that
    structure lives in `Feature.category` — in the registry, which is Python, and
    therefore invisible to anything querying the database. So it goes in a table
    beside the numbers.

    The SQL definition travels too. It is what makes a column answerable rather
    than merely present: `rsi_14` on a 14-session SIMPLE average is a different
    number from the exponential one a charting package draws, and reading the
    expression is the only way to know which you have.
    """
    _exec(
        con,
        f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.feature_catalog (
                feature     text PRIMARY KEY,
                dimension   text NOT NULL,
                rank_column text NOT NULL,
                lookback    int  NOT NULL,
                definition  text NOT NULL)""",
    )
    _exec(con, f"TRUNCATE {SCHEMA}.feature_catalog")
    _exec(con, _catalog_insert_sql())
    return con.execute(f"SELECT count(*) FROM pg.{SCHEMA}.feature_catalog").fetchone()[0]


def _catalog_insert_sql() -> str:
    """Built from the registry, so a feature added tomorrow is catalogued by
    existing rather than by anyone remembering to list it here."""
    from vnresearch.features.registry import all_features

    rows = ", ".join(
        f"({_literal(f.name)}, {_literal(f.category)}, {_literal(f.name + '_rank')}, "
        f"{f.lookback}, {_literal(f.sql)})"
        for f in sorted(all_features().values(), key=lambda x: (x.category, x.name))
    )
    return (
        f"INSERT INTO {SCHEMA}.feature_catalog "
        f"(feature, dimension, rank_column, lookback, definition) VALUES {rows}"
    )


def _record_run(con, rows: dict[str, int]) -> None:
    cfg = config.load("model")["dataset"]
    h = config.load("features")["label"]["horizon"]
    try:
        as_of = load_manifest(config.path(config.load("data")["mirror_dir"])).as_of
    except FileNotFoundError:
        as_of = None

    max_date = con.execute(f"SELECT max(date) FROM pg.{SCHEMA}.features").fetchone()[0]
    vals = (
        f"'{as_of}'" if as_of else "NULL",
        str(rows["features"]),
        str(rows["labels"]),
        str(rows["features_cols"]),
        str(h),
        f"'{cfg.get('target', 'residual')}'",
        f"DATE '{cfg['start_date']}'" if cfg.get("start_date") else "NULL",
        f"DATE '{cfg['holdout_start']}'" if cfg.get("holdout_start") else "NULL",
        f"DATE '{max_date}'",
    )
    _exec(
        con,
        f"""INSERT INTO {SCHEMA}.publish_runs
            (mirror_as_of, features_rows, labels_rows, feature_cols,
             horizon, target, start_date, holdout_start, last_session)
            VALUES ({", ".join(vals)})""",
    )


def build(verbose: bool = True) -> dict[str, Any]:
    """Write the research schema. Full replace, not an upsert.

    The pipeline rebuilds every parquet from the mirror on each run, so a
    partial update would leave rows from a previous feature set sitting beside
    the current ones with nothing to mark which was which.
    """
    feats_path = config.path("data/features/features.parquet")
    labels_path = config.path("data/clean/labels.parquet")
    for p in (feats_path, labels_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run `vnr pipeline` first")

    con = _connect()
    try:
        _exec(con, f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
        # Views depend on the tables, and Postgres refuses to drop a table out
        # from under one. They are rebuilt from config at the end anyway.
        for v in ("training_sample", "holdout_sample"):
            _exec(con, f"DROP VIEW IF EXISTS {SCHEMA}.{v}")

        t0 = time.time()
        cols = _feature_columns(con, feats_path.as_posix())
        con.execute(
            f"CREATE OR REPLACE TABLE pg.{SCHEMA}.features AS "
            f"SELECT {', '.join(cols)} FROM read_parquet('{feats_path.as_posix()}')"
        )
        n_feat = con.execute(f"SELECT count(*) FROM pg.{SCHEMA}.features").fetchone()[0]
        if verbose:
            took = time.time() - t0
            print(f"  features            {n_feat:>9,} rows x {len(cols):>3} cols  {took:5.1f}s")

        # Universe rows only, which is not a reduction in what is usable: the
        # feature table is universe-only by construction, so the other 3.8M
        # label rows have nothing to join to and nothing downstream reads them.
        t0 = time.time()
        con.execute(
            f"CREATE OR REPLACE TABLE pg.{SCHEMA}.labels AS "
            f"SELECT * FROM read_parquet('{labels_path.as_posix()}') WHERE in_universe"
        )
        n_lab = con.execute(f"SELECT count(*) FROM pg.{SCHEMA}.labels").fetchone()[0]
        if verbose:
            print(
                f"  labels              {n_lab:>9,} rows                 {time.time() - t0:5.1f}s"
            )

        # UNIQUE, not just an index: (ticker, date) is the key both tables are
        # joined on, and a duplicate would silently multiply the training set.
        # Named, so a run that died between the table and its index can be
        # repeated instead of colliding with an auto-generated name.
        for t in ("features", "labels"):
            _exec(con, f"CREATE UNIQUE INDEX IF NOT EXISTS {t}_key ON {SCHEMA}.{t} (ticker, date)")
            _exec(con, f"CREATE INDEX IF NOT EXISTS {t}_date ON {SCHEMA}.{t} (date)")

        _exec(
            con,
            f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.publish_runs (
                    id            bigserial PRIMARY KEY,
                    published_at  timestamptz NOT NULL DEFAULT now(),
                    mirror_as_of  text,
                    features_rows bigint,
                    labels_rows   bigint,
                    feature_cols  int,
                    horizon       int,
                    target        text,
                    start_date    date,
                    holdout_start date,
                    last_session  date)""",
        )

        for name, holdout_only in (("training_sample", False), ("holdout_sample", True)):
            _exec(con, _sample_view_sql(name, holdout_only))
        n_train = con.execute(f"SELECT count(*) FROM pg.{SCHEMA}.training_sample").fetchone()[0]
        n_hold = con.execute(f"SELECT count(*) FROM pg.{SCHEMA}.holdout_sample").fetchone()[0]

        n_cat = _write_catalog(con)
        _record_run(con, {"features": n_feat, "labels": n_lab, "features_cols": len(cols)})
        _comment(con)

        if verbose:
            dims = con.execute(
                f"SELECT dimension, count(*) FROM pg.{SCHEMA}.feature_catalog GROUP BY 1 ORDER BY 1"
            ).fetchall()
            print(f"  training_sample     {n_train:>9,} rows   (view, holdout excluded)")
            print(f"  holdout_sample      {n_hold:>9,} rows   (view, frozen — see invariant 10)")
            print(f"  feature_catalog     {n_cat:>9} features across {len(dims)} dimensions")
            print("                        " + ", ".join(f"{d} {n}" for d, n in dims))
        return {
            "features": n_feat,
            "labels": n_lab,
            "training_sample": n_train,
            "holdout_sample": n_hold,
            "feature_catalog": n_cat,
        }
    finally:
        con.close()


_COMMENTS: tuple[tuple[str, str, str], ...] = (
    (
        "TABLE",
        "features",
        (
            "Feature matrix from vnstock-research, universe rows only. Feature columns "
            "are REAL because the model reads them as float32. Rebuild with `vnr publish`."
        ),
    ),
    (
        "TABLE",
        "labels",
        (
            "Forward returns with tradeable entry, per horizon. Entry is the NEXT "
            "session's open and must be tradeable; exit is deliberately not required to be."
        ),
    ),
    (
        "TABLE",
        "feature_catalog",
        (
            "One row per feature: which of the 8 dimensions it belongs to, how much "
            "history it needs, and the SQL that defines it. Rebuilt from the registry "
            "on every publish, so it cannot drift from the columns in features."
        ),
    ),
    (
        "TABLE",
        "publish_runs",
        (
            "One row per `vnr publish`. Compare mirror_as_of against "
            "data/mirror/manifest.json to tell whether this copy is current."
        ),
    ),
    (
        "VIEW",
        "training_sample",
        (
            "What the model trains on: features joined to labels, filtered and targeted "
            "per config/model.yaml. Holdout excluded."
        ),
    ),
    (
        "VIEW",
        "holdout_sample",
        "The frozen holdout. Do not train on it or tune against it.",
    ),
)


def _comment(con) -> None:
    """Tell someone reading the database what these are and where they came from."""
    for kind, name, text in _COMMENTS:
        _exec(con, f"COMMENT ON {kind} {SCHEMA}.{name} IS {_literal(text)}")
