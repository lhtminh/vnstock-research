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
def features() -> None:
    """Compute every registered feature plus its cross-sectional rank."""
    from vnresearch.features import build as fbuild

    typer.echo(f"\n-> {fbuild.build()}")


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

    out = config.path("data") / f"oos_{model}.parquet"
    run.oos.to_parquet(out)
    typer.echo(f"  -> {out}")

    typer.echo("\n  top features:")
    for name, val in run.importance.head(8).items():
        typer.echo(f"    {name:<26} {val:8.1f}")

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
    """Run mirror -> clean -> panel -> label -> features end to end."""
    for step in (mirror, clean, panel, label, features):
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
