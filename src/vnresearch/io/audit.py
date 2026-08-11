"""Data-quality checks against the published `research` schema.

WHY THIS EXISTS. Every defect this repo has shipped was invisible until someone
went looking: a NULL ranking above 83% of the market, a NaN scoring as the most
extreme name in the market, a 2026 share count applied to a 2010 bar. None of
them raised anything. The test suite proves the CODE is right on synthetic data;
this asks whether the DATA in front of you is sound, which is a different
question and the one a portfolio manager is actually exposed to.

Results are written to `research.quality_checks`, one row per check per run, so
"was the database clean on the day we traded" is answerable after the fact. A
check that only prints is a check nobody can cite.

Every check names what it found rather than returning a bare boolean —
`n_bad = 177` and a sample of the offending keys, so the next step is obvious.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vnresearch import config
from vnresearch.io.publish import SCHEMA, _connect, _exec, _literal


@dataclass
class Check:
    name: str
    scope: str
    detail: str
    n_bad: int = 0
    sample: str = ""
    passed: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        self.passed = self.n_bad == 0


def _numeric_columns(con, table: str) -> list[str]:
    return [
        r[0]
        for r in con.execute(
            f"""SELECT column_name FROM pg.information_schema.columns
                WHERE table_schema = '{SCHEMA}' AND table_name = '{table}'
                  AND data_type IN ('real', 'double precision')
                ORDER BY ordinal_position"""
        ).fetchall()
    ]


def _nonfinite(con, table: str) -> list[Check]:
    """NaN or infinity anywhere in a numeric column.

    The check that would have caught the rank-1.0 defect. NaN survives every
    `IS NOT NULL` guard, so it reaches ranking, averaging and thresholding as
    though it were a number — and in a percentile rank it wins.
    """
    cols = _numeric_columns(con, table)
    if not cols:
        return []
    parts = ", ".join(f"count(*) FILTER (WHERE isnan({c}) OR isinf({c})) AS {c}" for c in cols)
    row = con.execute(f"SELECT {parts} FROM pg.{SCHEMA}.{table}").fetchone()
    bad = [(c, int(v)) for c, v in zip(cols, row) if v]
    return [
        Check(
            name="no_nan_or_inf",
            scope=f"{table}.{c}",
            detail="NaN/Inf is not NULL: it passes every missing-value guard and ranks as a value",
            n_bad=n,
        )
        for c, n in bad
    ]


def _key_checks(con, table: str) -> list[Check]:
    """(ticker, date) must be present and unique — everything joins on it."""
    out = []
    n = con.execute(
        f"SELECT count(*) FROM pg.{SCHEMA}.{table} WHERE ticker IS NULL OR date IS NULL"
    ).fetchone()[0]
    out.append(
        Check("key_not_null", table, "a row with no key cannot be joined or corrected", int(n))
    )
    dup = con.execute(
        f"""SELECT count(*) FROM (SELECT ticker, date FROM pg.{SCHEMA}.{table}
            GROUP BY 1, 2 HAVING count(*) > 1)"""
    ).fetchone()[0]
    out.append(
        Check("key_unique", table, "a duplicate key silently multiplies every joined row", int(dup))
    )
    return out


def _rank_range(con) -> list[Check]:
    """Every *_rank must be a percentile in [0, 1] or absent."""
    cols = [c for c in _numeric_columns(con, "features") if c.endswith("_rank")]
    parts = ", ".join(f"count(*) FILTER (WHERE {c} < 0 OR {c} > 1) AS {c}" for c in cols)
    row = con.execute(f"SELECT {parts} FROM pg.{SCHEMA}.features").fetchone()
    return [
        Check(
            "rank_in_unit_interval",
            f"features.{c}",
            "a percentile outside [0,1] is not one",
            int(v),
        )
        for c, v in zip(cols, row)
        if v
    ]


def _enumerations(con, tables: set[str]) -> list[Check]:
    out = []
    labels = config.load("speculation")["labels"]
    allowed = ", ".join(_literal(x) for x in labels)
    if "speculation" in tables:
        for col in ("spec_label", "spec_label_adj", "pvdi_label", "turnover_label"):
            n = con.execute(
                f"""SELECT count(*) FROM pg.{SCHEMA}.speculation
                    WHERE {col} IS NOT NULL AND {col} NOT IN ({allowed})"""
            ).fetchone()[0]
            out.append(
                Check("label_in_enum", f"speculation.{col}", f"one of: {', '.join(labels)}", int(n))
            )
    if "ratings" in tables:
        from vnresearch.rating.score import GRADES

        grades = ", ".join(_literal(g) for g, _ in GRADES)
        n = con.execute(
            f"""SELECT count(*) FROM pg.{SCHEMA}.ratings
                WHERE rating IS NOT NULL AND rating NOT IN ({grades})"""
        ).fetchone()[0]
        out.append(Check("label_in_enum", "ratings.rating", f"one of: {grades}", int(n)))
    return out


def _referential(con, tables: set[str]) -> list[Check]:
    """The research tables against each other and against the service's own.

    Not declared as foreign keys on purpose. A FK from `research` into `public`
    would make the service's migrations depend on this schema, and the whole
    point of the split is that they do not. Checked instead of enforced.
    """
    out = []
    n = con.execute(
        f"""SELECT count(*) FROM pg.{SCHEMA}.features f
            LEFT JOIN pg.public.symbols s ON s.ticker = f.ticker
            WHERE s.ticker IS NULL"""
    ).fetchone()[0]
    out.append(
        Check(
            "ticker_known_to_service", "features", "every ticker exists in public.symbols", int(n)
        )
    )

    n = con.execute(
        f"""SELECT count(*) FROM pg.{SCHEMA}.features f
            LEFT JOIN pg.{SCHEMA}.labels l ON l.ticker = f.ticker AND l.date = f.date
            WHERE l.ticker IS NULL"""
    ).fetchone()[0]
    out.append(
        Check("features_have_labels", "features->labels", "the training join loses no row", int(n))
    )

    if "ratings" in tables:
        n = con.execute(
            f"""SELECT count(*) FROM pg.{SCHEMA}.features f
                LEFT JOIN pg.{SCHEMA}.ratings r ON r.ticker = f.ticker AND r.date = f.date
                WHERE r.ticker IS NULL"""
        ).fetchone()[0]
        out.append(
            Check(
                "every_name_is_rated",
                "features->ratings",
                "a name with no rating is invisible to a screen that filters on grade",
                int(n),
            )
        )
    return out


def _catalog_matches_columns(con) -> list[Check]:
    """The dimension metadata must describe columns that exist, and vice versa."""
    orphan = con.execute(
        f"""SELECT count(*) FROM pg.{SCHEMA}.feature_catalog c
            WHERE NOT EXISTS (SELECT 1 FROM pg.information_schema.columns x
                              WHERE x.table_schema = '{SCHEMA}' AND x.table_name = 'features'
                                AND x.column_name = c.rank_column)"""
    ).fetchone()[0]
    uncat = con.execute(
        f"""SELECT count(*) FROM pg.information_schema.columns x
            WHERE x.table_schema = '{SCHEMA}' AND x.table_name = 'features'
              AND x.column_name LIKE '%\\_rank'
              AND NOT EXISTS (SELECT 1 FROM pg.{SCHEMA}.feature_catalog c
                              WHERE c.rank_column = x.column_name)"""
    ).fetchone()[0]
    return [
        Check(
            "catalog_has_no_orphans",
            "feature_catalog",
            "a dimension for a column that does not exist is metadata with nothing under it",
            int(orphan),
        ),
        Check(
            "every_rank_is_catalogued",
            "features",
            "an uncatalogued column has no dimension and no definition",
            int(uncat),
        ),
    ]


def _no_dead_columns(con) -> list[Check]:
    """A feature that is NULL everywhere is a bug, not a thin one."""
    cols = [c for c in _numeric_columns(con, "features") if not c.endswith("_rank")]
    parts = ", ".join(f"count({c}) AS {c}" for c in cols)
    row = con.execute(f"SELECT {parts} FROM pg.{SCHEMA}.features").fetchone()
    return [
        Check(
            "feature_has_some_values",
            f"features.{c}",
            "all-NULL means the formula never resolved, not that the data is thin",
            1,
        )
        for c, v in zip(cols, row)
        if v == 0
    ]


def _universe(con) -> list[Check]:
    """The liquidity floor, and that it is applied POINT-IN-TIME.

    The floor itself is easy to check and easy to get wrong silently — a broken
    universe build would quietly admit names nobody can trade.

    The second check is the one that matters more and is easy to lose. Universe
    membership is decided per DAY, from a trailing average known on that day. A
    per-TICKER screen on lifetime average liquidity looks tidier and is
    forward-looking: deciding whether a 2010 row exists needs 2026 data, which
    keeps precisely the names that did not later collapse. Measured on this
    sample, rows a lifetime-1bn screen would keep earned +0.24% a month
    market-relative against -1.49% for the ones it would drop — about 5% a year
    of pure illusion, on rows that had ALREADY passed the daily test.

    So this asserts the universe still churns. A membership set that stopped
    changing would mean someone had frozen it, and a frozen set is a screen on
    hindsight whatever it is called.
    """
    # float() defensively: YAML 1.1 loads an unsigned exponent like `1.0e9` as a
    # string, and this reader should not depend on how the number was written.
    floor = float(config.load("features")["universe"]["adtv_min_vnd"])
    n = con.execute(f"SELECT count(*) FROM pg.{SCHEMA}.features WHERE adtv < {floor}").fetchone()[0]
    out = [
        Check(
            "universe_meets_liquidity_floor",
            "features.adtv",
            f"every row must clear {floor:,.0f} VND on its own date",
            int(n),
        )
    ]
    # Entrants and leavers over the last year. Zero of both across a whole year
    # of a market this size means membership is no longer being recomputed.
    churn = con.execute(
        f"""WITH d AS (SELECT ticker, min(date) a, max(date) b
                       FROM pg.{SCHEMA}.features GROUP BY 1),
                 lim AS (SELECT max(date) - 365 AS cut FROM pg.{SCHEMA}.features)
            SELECT count(*) FILTER (WHERE a > lim.cut) + count(*) FILTER (WHERE b < lim.cut)
            FROM d, lim"""
    ).fetchone()[0]
    out.append(
        Check(
            "universe_still_churns",
            "features",
            "a membership set that never changes is a screen on hindsight",
            0 if churn else 1,
        )
    )
    return out


def _freshness(con) -> list[Check]:
    """Does the published copy match the mirror it claims to come from."""
    from vnresearch.io.manifest import load_manifest

    try:
        as_of = load_manifest(config.path(config.load("data")["mirror_dir"])).as_of
    except FileNotFoundError:
        return []
    row = con.execute(
        f"SELECT mirror_as_of FROM pg.{SCHEMA}.publish_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    stale = 0 if row and row[0] == as_of else 1
    return [
        Check(
            "published_copy_is_current",
            "publish_runs",
            f"local mirror {as_of}, last publish {row[0] if row else 'never'} — "
            f"run `vnr publish` if these differ",
            stale,
        )
    ]


def run(verbose: bool = True) -> list[Check]:
    """Every check, recorded to research.quality_checks."""
    con = _connect()
    try:
        tables = {
            r[0]
            for r in con.execute(
                f"""SELECT table_name FROM pg.information_schema.tables
                    WHERE table_schema = '{SCHEMA}'"""
            ).fetchall()
        }
        if "features" not in tables:
            raise RuntimeError(f"schema {SCHEMA} has no features table — run `vnr publish` first")

        checks: list[Check] = []
        for t in ("features", "labels", "speculation", "ratings"):
            if t in tables:
                checks += _key_checks(con, t) + _nonfinite(con, t)
        checks += _rank_range(con)
        checks += _enumerations(con, tables)
        checks += _referential(con, tables)
        checks += _catalog_matches_columns(con)
        checks += _no_dead_columns(con)
        checks += _universe(con)
        checks += _freshness(con)

        _exec(
            con,
            f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.quality_checks (
                    id       bigserial PRIMARY KEY,
                    ran_at   timestamptz NOT NULL DEFAULT now(),
                    check_name text NOT NULL,
                    scope    text NOT NULL,
                    passed   boolean NOT NULL,
                    n_bad    bigint NOT NULL,
                    detail   text)""",
        )
        rows = ", ".join(
            f"({_literal(c.name)}, {_literal(c.scope)}, {c.passed}, {c.n_bad}, "
            f"{_literal(c.detail)})"
            for c in checks
        )
        if rows:
            _exec(
                con,
                f"INSERT INTO {SCHEMA}.quality_checks (check_name, scope, passed, n_bad, detail) "
                f"VALUES {rows}",
            )
        if verbose:
            _report(checks)
        return checks
    finally:
        con.close()


def _report(checks: list[Check]) -> None:
    failed = [c for c in checks if not c.passed]
    print(f"  {len(checks) - len(failed)}/{len(checks)} checks passed")
    if not failed:
        print("  research schema is clean")
        return
    print("\n  FAILED:")
    for c in failed:
        print(f"    {c.name:<26} {c.scope:<34} {c.n_bad:>9,}")
        print(f"      {c.detail}")
