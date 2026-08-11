"""Command line entry point: `vnr <command>`."""

from __future__ import annotations

import typer

from vnresearch import config
from vnresearch.io import duck, manifest

app = typer.Typer(add_completion=False, help="Vietnamese market alpha research pipeline.")


@app.command()
def mirror() -> None:
    """Copy the Postgres tables to the local Parquet mirror."""
    typer.echo(f"mirroring from {config.dsn().split('@')[-1]}")
    m = duck.mirror()
    m.check_single_epoch()
    typer.echo(f"\nas_of        {m.as_of}")
    typer.echo(f"epochs       {m.adjustment_epochs}")
    typer.echo(f"last ingest  {m.max_ingested_at}")


@app.command()
def clean() -> None:
    """Rebuild bar_status (the source stores it as NULL) into data/clean/bars.parquet."""
    from vnresearch.clean import bars

    path = bars.build()
    typer.echo(f"\n-> {path}")


@app.command()
def panel() -> None:
    """Build the base panel: tradeability, liquidity, universe, market return."""
    from vnresearch.clean import panel as panel_mod

    typer.echo(f"\n-> {panel_mod.build()}")


@app.command()
def label() -> None:
    """Build forward-return labels with tradeable entry."""
    from vnresearch.label import forward

    typer.echo(f"\n-> {forward.build()}")


@app.command()
def peers() -> None:
    """Build the correlation-based peer sets the peer_* features read."""
    from vnresearch.features import peers as peers_mod

    typer.echo(f"\n-> {peers_mod.build()}")


@app.command()
def features() -> None:
    """Compute every registered feature plus its cross-sectional rank."""
    from vnresearch.features import build as fbuild

    typer.echo(f"\n-> {fbuild.build()}")


@app.command()
def rating(check: bool = False) -> None:
    """Score every name: a Z-score per dimension, a composite, and a grade.

    Explains rather than predicts — see rating/score.py. Needs `vnr features`,
    and picks up the speculation penalty if `vnr speculation` has been run.

    --check reports where a feature's DECLARED direction disagrees with what it
    actually did on the dev period. It reports; it does not flip anything.
    """
    from vnresearch.rating import score

    if check:
        df = score.check_directions()
        # Compared to False rather than negated: `agrees` is None where the IC
        # could not be measured at all, and `~df["agrees"]` would count those as
        # disagreements.
        bad = df[df["agrees"].eq(True) == False]
        bad = bad[bad["ic"].notna()]
        typer.echo(f"\n  {len(df) - len(bad)}/{len(df)} rated features agree with their sign\n")
        if len(bad):
            typer.echo("  DISAGREES — declared sign vs measured IC:")
            typer.echo(bad.to_string(index=False))
        return

    typer.echo(f"\n-> {score.build()}")


@app.command()
def speculation() -> None:
    """Label speculation per the mentor's document: PVDI, turnover, volatility, range.

    ~5 minutes — the percentile step pools every stock over a trailing year and
    recomputes it for each of 5,900 dates. This is a DESCRIPTION of how a stock
    is behaving, built from trailing data only, not a forward-looking target.
    """
    from vnresearch.label import speculation as spec

    typer.echo(f"\n-> {spec.build()}")


@app.command()
def publish() -> None:
    """Copy the labelled sample into Postgres, schema `research`.

    Deliberately NOT part of `vnr pipeline`. The pipeline runs on the clock
    before the 15:00 trading decision, and nothing on that path reads these
    tables — they are for querying from outside this repo. `research.publish_runs`
    records the mirror snapshot each copy came from, so staleness is visible.
    """
    from vnresearch.io import publish as pub

    typer.echo(f"publishing to {config.dsn().split('@')[-1]}, schema {pub.SCHEMA}")
    pub.build()


@app.command()
def audit() -> None:
    """Check the published `research` schema and record the result.

    Asks whether the DATA in front of you is sound, which the test suite does
    not — that proves the code is right on synthetic input. Results land in
    `research.quality_checks`, so "was the database clean the day we traded" is
    answerable afterwards. Exits non-zero if anything failed.
    """
    import sys

    from vnresearch.io import audit as au

    failed = [c for c in au.run() if not c.passed]
    if failed:
        sys.exit(1)


@app.command()
def alpha(horizon: int = 5, top: int = 10) -> None:
    """Measure each feature standalone and write a markdown report."""
    from vnresearch.alpha import report

    typer.echo(f"\n-> {report.build(horizon=horizon, top=top)}")


@app.command()
def train(model: str = "lightgbm", controls: bool = False) -> None:
    """Walk-forward training with purged folds. --controls runs the leakage checks."""
    from vnresearch.model import train as tr

    run = tr.walk_forward(model)
    typer.echo(f"\n  mean IC across folds  {run.mean_ic:+.4f}")
    # Printed next to IC because the two answer different questions and can
    # disagree: IC orders ~280 names, this one scores the 50 that get bought.
    # The 0.60% is the round trip a rebalance pays, so an edge under it is a
    # losing book no matter what the IC says.
    edge = run.mean_top_edge
    typer.echo(f"  top-50 gross edge     {edge:+.4f}  per holding period  (round trip 0.0060)")

    out = config.path("data") / f"oos_{model}.parquet"
    run.oos.to_parquet(out)
    typer.echo(f"  -> {out}")

    typer.echo("\n  top features:")
    for name, val in run.importance.head(8).items():
        typer.echo(f"    {name:<26} {val:8.1f}")

    # Archived, not just printed. Eight lines of stdout cannot answer "did the
    # feature I added last week displace anything", which is the only question
    # that matters after a registry change. Stamped with the run time so two
    # feature sets can be diffed directly.
    from datetime import UTC, datetime

    from vnresearch.features.registry import all_features

    reg = all_features()
    imp = run.importance_by_fold.copy()
    imp.insert(0, "gain_pct", run.importance)
    imp.insert(0, "category", [reg[n.removesuffix("_rank")].category for n in imp.index])
    imp.insert(0, "rank", range(1, len(imp) + 1))
    imp.index.name = "feature"
    reports = config.path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    imp_path = reports / f"importance-{model}-{stamp}.csv"
    imp.round(4).to_csv(imp_path)
    typer.echo(f"  -> {imp_path}")

    if controls:
        typer.echo("\n  leakage controls:")
        typer.echo(tr.leakage_controls(model).to_string(index=False))


@app.command()
def holdout(model: str = "lightgbm") -> None:
    """Score once on the frozen holdout. The only number development never saw."""
    from vnresearch.model import train as tr

    r = tr.evaluate_holdout(model)
    oos = r.pop("oos")
    for k, v in r.items():
        typer.echo(f"  {k:<16} {v:.4f}" if isinstance(v, float) else f"  {k:<16} {v}")

    out = config.path("data") / f"oos_{model}_holdout.parquet"
    oos.to_parquet(out)
    typer.echo(f"  -> {out}")


@app.command()
def freeze(model: str = "lightgbm", with_cv: bool = False) -> None:
    """Fit on the whole sample and save the artifact paper trading loads.

    Trains through the last labelled session, holdout INCLUDED — going live is
    what the holdout was being kept for. --with-cv also records walk-forward
    scores in the manifest (~7 min).
    """
    from vnresearch.model import freeze as fz

    fz.build(model, with_cv=with_cv)


@app.command()
def backtest(model: str = "lightgbm", compare: bool = False, holdout: bool = False) -> None:
    """Backtest out-of-sample predictions. --compare shows the untradeable-fill delta."""
    import pandas as pd

    from vnresearch.backtest import engine

    suffix = "_holdout" if holdout else ""
    path = config.path("data") / f"oos_{model}{suffix}.parquet"
    if not path.exists():
        cmd = "holdout" if holdout else "train"
        raise typer.BadParameter(f"{path} missing — run `vnr {cmd} --model {model}` first")
    oos = pd.read_parquet(path)

    typer.echo("\n=== tradeable fills only ===")
    honest = engine.run(oos)
    if not compare:
        return

    typer.echo("\n=== fills allowed on limit-locked bars ===")
    loose = engine.run(oos, allow_untradeable_fills=True)
    a = float(honest.stats["Total Return [%]"])
    b = float(loose.stats["Total Return [%]"])
    typer.echo(f"\n  phantom return removed by the filter: {b - a:+.1f} pp ({b / a - 1:+.1%})")


@app.command()
def pipeline() -> None:
    """Run mirror -> clean -> panel -> label -> peers -> features end to end."""
    # peers runs BEFORE features and inside the pipeline, not out of band. It
    # used to be neither: build.py substitutes NULL for all four peer_* features
    # when peers.parquet is missing, so a stale or absent file silently removed
    # four features from the model instead of failing.
    for step in (mirror, clean, panel, label, peers, features):
        typer.echo(f"\n=== {step.__name__} ===")
        step()


@app.command()
def info() -> None:
    """Show what the local mirror contains."""
    cfg = config.load("data")
    out = config.path(cfg["mirror_dir"])
    m = manifest.load_manifest(out)
    typer.echo(f"mirror   {out}")
    typer.echo(f"as_of    {m.as_of}")
    typer.echo(f"epochs   {m.adjustment_epochs}")
    for t, n in m.row_counts.items():
        typer.echo(f"  {t:<20} {n:>9,}")


if __name__ == "__main__":
    app()
