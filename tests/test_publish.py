"""Publishing the labelled sample to Postgres.

No database is needed. The view SQL is ordinary standard SQL, so DuckDB can run
it against synthetic tables in a schema of the same name — which tests the thing
that can actually be wrong (what the view selects, filters and ranks) rather
than DuckDB's ability to talk to Postgres.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from vnresearch import config
from vnresearch.alpha.dataset import feature_names
from vnresearch.features.registry import all_features
from vnresearch.io import publish
from vnresearch.model.dataset import _target_sql


@pytest.fixture
def sql_train():
    return publish._sample_view_sql("training_sample", holdout_only=False)


@pytest.fixture
def sql_hold():
    return publish._sample_view_sql("holdout_sample", holdout_only=True)


def test_nothing_reaches_the_service_schema(sql_train, sql_hold):
    """The boundary this whole design exists to keep.

    vnstock-service owns `public` and applies its own migrations from Go with
    unqualified DDL. Research output goes in its own schema so that separation
    is mechanical rather than a promise someone remembers to keep.
    """
    for sql in (sql_train, sql_hold, *(f"x {n} y" for _, n, _ in publish._COMMENTS)):
        assert "public." not in sql
    assert publish.SCHEMA != "public"


def test_the_split_is_the_configured_holdout_date(sql_train, sql_hold):
    """Complementary, not merely different: every row lands in exactly one."""
    start = config.load("model")["dataset"]["holdout_start"]
    assert f"date < DATE '{start}'" in sql_train
    assert f"date >= DATE '{start}'" in sql_hold


def test_the_target_is_not_a_second_copy(sql_train):
    """stress.py kept its own copy of the target expression and went on scoring
    the old one after it changed. The view imports it instead."""
    cfg = config.load("model")["dataset"]
    h = config.load("features")["label"]["horizon"]
    assert _target_sql(h, cfg.get("target", "residual")) in sql_train


def test_every_registered_feature_is_published(sql_train):
    """A feature added to the registry must reach the published matrix without
    anyone editing this module."""
    for name in feature_names(ranked=config.load("model")["dataset"]["use_ranked_features"]):
        assert name in sql_train


def _tables(con, dates, tickers, bench=0.01, label_ok=True):
    """Synthetic research.features / research.labels, shaped from the registry."""
    feats = sorted(all_features())
    rows = []
    for d in dates:
        for i, t in enumerate(tickers):
            row = {"ticker": t, "date": d}
            for j, f in enumerate(feats):
                row[f] = 0.1 * (i + 1) + 0.01 * j
                row[f"{f}_rank"] = (i + 1) / len(tickers)
            rows.append(row)
    con.execute("CREATE SCHEMA IF NOT EXISTS research")
    con.register("f_df", pd.DataFrame(rows))
    con.execute("CREATE OR REPLACE TABLE research.features AS SELECT * FROM f_df")

    lab = [
        {
            "ticker": t,
            "date": d,
            "fwd_ret_5": 0.02 * (i + 1),
            "bench_ret_5": bench,
            "label_ok_5": label_ok,
        }
        for d in dates
        for i, t in enumerate(tickers)
    ]
    con.register("l_df", pd.DataFrame(lab))
    con.execute("CREATE OR REPLACE TABLE research.labels AS SELECT * FROM l_df")


def test_the_view_runs_and_ranks_within_the_day(sql_train):
    """y must be a percentile of the day's own cross-section. Ranking across
    days instead would compare a 2020 name against a 2015 one, which is
    invariant 4 one level down from the features."""
    con = duckdb.connect()
    dates = [pd.Timestamp("2020-03-02").date(), pd.Timestamp("2020-03-03").date()]
    _tables(con, dates, ["AAA", "BBB", "CCC", "DDD"])
    con.execute(sql_train)
    df = con.execute("SELECT * FROM research.training_sample").df()

    assert len(df) == 8
    for _, g in df.groupby("date"):
        assert g["y"].min() == 0.0
        assert g["y"].max() == 1.0
    con.close()


def test_a_row_with_no_computable_target_never_appears(sql_train):
    """A NULL benchmark makes the target NULL. Letting the row through would
    percentile-rank a missing value alongside real ones — invariant 3, and the
    same trap as the NULL-rank defect in the feature ranks."""
    con = duckdb.connect()
    dates = [pd.Timestamp("2020-03-02").date()]
    _tables(con, dates, ["AAA", "BBB", "CCC"], bench=None)
    con.execute(sql_train)
    assert con.execute("SELECT count(*) FROM research.training_sample").fetchone()[0] == 0
    con.close()


def test_an_unlabelled_row_never_appears(sql_train):
    """label_ok_5 is false when entry was untradeable or the exit gap was too
    wide. Those returns are fiction and must not be trained on."""
    con = duckdb.connect()
    dates = [pd.Timestamp("2020-03-02").date()]
    _tables(con, dates, ["AAA", "BBB", "CCC"], label_ok=False)
    con.execute(sql_train)
    assert con.execute("SELECT count(*) FROM research.training_sample").fetchone()[0] == 0
    con.close()


def test_the_holdout_view_excludes_the_development_years(sql_hold):
    con = duckdb.connect()
    dates = [pd.Timestamp("2020-03-02").date(), pd.Timestamp("2024-06-03").date()]
    _tables(con, dates, ["AAA", "BBB", "CCC"])
    con.execute(sql_hold)
    got = con.execute("SELECT DISTINCT date FROM research.holdout_sample").df()
    assert len(got) == 1
    assert got["date"].iloc[0].year == 2024
    con.close()


def test_features_narrow_to_real_and_prices_do_not(tmp_path):
    """REAL is only safe because dataset.load() casts to float32 anyway. That
    reasoning covers features; it does not cover a price or a traded value, so
    those must stay DOUBLE."""
    # Full width, because _feature_columns now also checks the file carries
    # every registered feature — a one-feature fixture is a file no publish
    # would accept.
    feats = sorted(all_features())
    cells = ["'AAA' AS ticker", "DATE '2020-01-02' AS date"]
    cells += [f"{v}::DOUBLE AS {c}" for v, c in ((1.5, "close"), (2.5, "adtv"))]
    cells += [f"3.5::DOUBLE AS {n}" for f in feats for n in (f, f"{f}_rank")]

    path = tmp_path / "f.parquet"
    con = duckdb.connect()
    con.execute(f"CREATE TABLE t AS SELECT {', '.join(cells)}")
    con.execute(f"COPY t TO '{path.as_posix()}' (FORMAT parquet)")

    cols = publish._feature_columns(con, path.as_posix())
    for f in feats:
        assert f"{f}::REAL AS {f}" in cols
        assert f"{f}_rank::REAL AS {f}_rank" in cols
    assert "close" in cols and "close::REAL AS close" not in cols
    assert "adtv" in cols and "adtv::REAL AS adtv" not in cols
    con.close()


def test_the_catalog_names_every_feature_and_its_dimension():
    """127 unlabelled float columns is not a feature set anyone can read. The
    dimension lives in `Feature.category`, which is Python and so invisible from
    SQL — the catalog is what carries it across."""
    sql = publish._catalog_insert_sql()
    feats = all_features()
    for name, f in feats.items():
        assert f"'{name}'" in sql
        assert f"'{name}_rank'" in sql
        assert f"'{f.category}'" in sql
    assert sql.count("), (") == len(feats) - 1


def test_the_catalog_round_trips_through_a_real_insert():
    """Definitions are SQL and contain quotes. An unescaped one closes the
    literal early, and the failure is a syntax error a long way from the cause."""
    con = duckdb.connect()
    con.execute("CREATE SCHEMA research")
    con.execute(
        """CREATE TABLE research.feature_catalog (
               feature text PRIMARY KEY, dimension text, rank_column text,
               lookback int, direction int, definition text)"""
    )
    con.execute(publish._catalog_insert_sql())
    got = con.execute("SELECT feature, dimension, definition FROM research.feature_catalog").df()

    feats = all_features()
    assert len(got) == len(feats)
    assert set(got["dimension"]) == {f.category for f in feats.values()}
    for _, row in got.iterrows():
        assert row["definition"] == feats[row["feature"]].sql
    con.close()


def test_every_catalogued_feature_reaches_the_training_matrix(sql_train):
    """The catalog describing a column the model never sees would be worse than
    no catalog — it would describe a feature set that does not exist."""
    con = duckdb.connect()
    con.execute("CREATE SCHEMA research")
    con.execute(
        """CREATE TABLE research.feature_catalog (
               feature text PRIMARY KEY, dimension text, rank_column text,
               lookback int, direction int, definition text)"""
    )
    con.execute(publish._catalog_insert_sql())
    ranks = con.execute("SELECT rank_column FROM research.feature_catalog").df()["rank_column"]
    for col in ranks:
        assert col in sql_train
    con.close()


def test_a_catalog_that_outruns_the_parquet_is_refused():
    """The catalog comes from the registry; the table comes from a parquet built
    by an earlier pipeline run. Add a feature, publish without rebuilding, and
    the catalog names a dimension for a column that does not exist — metadata
    with nothing under it, which reads as true. No database constraint can catch
    this: features are COLUMNS and a foreign key relates rows."""
    complete = {n for name in all_features() for n in (name, f"{name}_rank")}
    publish.check_registry_matches(complete)  # the matching case must not raise

    stale = complete - {"rsi_14_rank"}
    with pytest.raises(ValueError, match="rsi_14_rank"):
        publish.check_registry_matches(stale)


def test_the_refusal_says_which_command_fixes_it():
    """A failure that names the problem and not the remedy gets worked around."""
    with pytest.raises(ValueError, match="vnr pipeline"):
        publish.check_registry_matches(set())


def test_comment_text_with_an_apostrophe_stays_valid_sql():
    """The comments describe entry as the next session's open. Passing that
    through unescaped closes the string literal early and the DDL fails."""
    out = publish._literal("the next session's open")
    assert out == "'the next session''s open'"
    assert duckdb.connect().execute(f"SELECT {out}").fetchone()[0] == "the next session's open"
