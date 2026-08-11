"""Ratings: dimension Z-scores, the composite, and the grade.

The rating exists to be EXPLAINED, so these tests pin the things a portfolio
manager would be told: that a score is relative to the day, that a missing
feature is not quietly treated as average, and that the grade boundaries are
what the documentation says.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from vnresearch import config
from vnresearch.features.registry import all_features
from vnresearch.rating import score


def test_every_feature_declares_a_usable_direction():
    for f in all_features().values():
        assert f.direction in (-1, 0, 1), f.name


def test_only_signed_features_reach_the_rating():
    """Direction 0 means "carried, not rated". A volume spike has no sign until
    you know which way it broke, and averaging it in would add noise while
    looking like information."""
    rated = {c for feats in score.dimensions().values() for c, _ in feats}
    for f in all_features().values():
        assert (f"{f.name}_rank" in rated) == bool(f.direction), f.name


def test_season_is_carried_but_never_rated():
    """A calendar tendency is not a quality judgement — seas_tet says when a
    stock tends to move, not whether it is a good holding."""
    assert "season" not in score.dimensions()
    assert all(f.direction == 0 for f in all_features().values() if f.category == "season")


def test_every_rated_dimension_has_a_weight():
    weights = config.load("rating")["weights"]
    for dim in score.dimensions():
        assert dim in weights, dim


def test_a_missing_weight_is_refused_rather_than_defaulted(tmp_path, monkeypatch):
    """Silently defaulting a new dimension to 0 would drop it from every rating
    with nothing to show that it happened."""
    real = config.load

    def fake(name):
        cfg = real(name)
        if name == "rating":
            cfg = {**cfg, "weights": {k: v for k, v in cfg["weights"].items() if k != "momentum"}}
        return cfg

    monkeypatch.setattr(config, "load", fake)
    monkeypatch.setattr(config, "path", lambda rel: tmp_path / "features.parquet")
    (tmp_path / "features.parquet").write_bytes(b"")
    with pytest.raises(ValueError, match="momentum"):
        score.build(verbose=False)


def test_the_z_score_is_taken_within_the_day():
    """Invariant 4, one layer up: a stock is scored against the names trading
    that day, never against 2015. Scoring across the pooled sample would rate
    every name in a calm year highly and every name in a crisis poorly."""
    con = duckdb.connect()
    con.execute(
        """CREATE TABLE f AS SELECT * FROM (VALUES
             ('AAA', DATE '2020-01-02', 1.0), ('BBB', DATE '2020-01-02', 3.0),
             ('AAA', DATE '2020-01-03', 10.0), ('BBB', DATE '2020-01-03', 30.0)
           ) v(ticker, date, x)"""
    )
    got = con.execute(
        f"SELECT ticker, date, {score._z('x')} AS z FROM f ORDER BY date, ticker"
    ).df()
    # Same shape both days despite a 10x level shift — that is what "within the
    # day" buys, and a pooled Z would not give it.
    assert got["z"].iloc[0] == pytest.approx(got["z"].iloc[2])
    assert got["z"].iloc[1] == pytest.approx(got["z"].iloc[3])
    assert got["z"].iloc[0] < 0 < got["z"].iloc[1]
    con.close()


def test_a_day_with_no_spread_gives_no_z_rather_than_zero():
    """When every name shares a value the Z-score is undefined, which is not the
    same as it being average. NULLIF, not a fabricated floor."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE f AS SELECT * FROM (VALUES "
        "('AAA', DATE '2020-01-02', 5.0), ('BBB', DATE '2020-01-02', 5.0)) v(ticker, date, x)"
    )
    got = con.execute(f"SELECT {score._z('x')} AS z FROM f").df()
    assert got["z"].isna().all()
    con.close()


def test_the_grades_partition_the_day():
    """A/B/C/D/E must cover the universe exactly once, best first."""
    letters = [g for g, _ in score.GRADES]
    cuts = [p for _, p in score.GRADES]
    assert letters == ["A", "B", "C", "D", "E"]
    assert cuts == sorted(cuts), "boundaries must ascend or the CASE short-circuits wrongly"
    assert cuts[-1] == 1.0, "the last grade must absorb the rest of the universe"
    assert cuts[0] == pytest.approx(0.10)


def _tiny_features(tmp_path):
    """A synthetic feature file wide enough for build() to run."""
    feats = sorted(all_features())
    rows = []
    for d in (pd.Timestamp("2020-03-02").date(), pd.Timestamp("2020-03-03").date()):
        for i, t in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"]):
            row = {
                "ticker": t,
                "date": d,
                "close": 10_000.0 + i,
                "adtv": 1e9,
                "exchange": "HOSE",
                "in_universe": True,
            }
            for j, f in enumerate(feats):
                row[f] = 0.1 * i + 0.01 * j
                row[f"{f}_rank"] = (i + 1) / 5
            rows.append(row)
    path = tmp_path / "features.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_build_produces_one_row_per_name_and_a_grade(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config,
        "path",
        lambda rel: (
            tmp_path / "features.parquet"
            if rel.endswith("features.parquet")
            else tmp_path / "out.parquet"
        ),
    )
    _tiny_features(tmp_path)
    out = score.build(verbose=False)
    df = pd.read_parquet(out)

    assert len(df) == 10
    assert set(df["rating"]) <= {"A", "B", "C", "D", "E"}
    for dim in score.dimensions():
        assert f"z_{dim}" in df.columns
    assert "z_composite" in df.columns and "rating_score" in df.columns
    # No speculation file in tmp_path, so the penalty is absent and the score is
    # the composite untouched — not a silently zeroed column.
    assert df["spec_score"].isna().all()
    assert (df["rating_score"] - df["z_composite"]).abs().max() == pytest.approx(0)


def test_a_higher_composite_never_gets_a_worse_grade(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config,
        "path",
        lambda rel: (
            tmp_path / "features.parquet"
            if rel.endswith("features.parquet")
            else tmp_path / "out.parquet"
        ),
    )
    _tiny_features(tmp_path)
    df = pd.read_parquet(score.build(verbose=False))
    order = {g: i for i, (g, _) in enumerate(score.GRADES)}
    for _, day in df.groupby("date"):
        day = day.sort_values("rating_score", ascending=False)
        ranks = [order[g] for g in day["rating"]]
        assert ranks == sorted(ranks), "grade must be monotone in the score"
