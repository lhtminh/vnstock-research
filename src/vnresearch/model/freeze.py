"""Freeze a model to disk so something outside this repo can trade with it.

Everywhere else here, a model is fitted and thrown away — `walk_forward` and
`evaluate_holdout` both measure a PROCEDURE, and the estimator is a by-product.
Paper trading needs the opposite: one specific fitted object, kept, so that a
score produced next Tuesday came from the same model as last Tuesday's.

WHAT IT TRAINS ON. Everything from `start_date` through the last labelled
session, holdout included. Going live is what the holdout was being saved for;
its verdict is already recorded and does not un-happen. Withholding the most
recent 2.5 years from a model about to trade tomorrow buys nothing and costs the
regime it will actually meet.

THE MANIFEST IS NOT DECORATION. Every field is there because its absence lets a
silent mismatch through, and the loudest one is the feature list: LightGBM takes
a positional matrix, so a reordered X scores confidently and wrongly without
ever raising. `FrozenModel.predict` refuses rather than reorders on trust.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vnresearch import config
from vnresearch.model import dataset as ds
from vnresearch.model import train as tr

MODEL_DIR = "models/frozen"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _versions() -> dict[str, str]:
    """Library versions, because a model is only reproducible alongside them.

    A LightGBM booster pickled by one version and unpickled by another can load
    without error and predict differently. Recording the versions does not
    prevent that; it makes it diagnosable instead of mysterious.
    """
    import lightgbm
    import sklearn

    out = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
    }
    try:
        import xgboost

        out["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    return out


def _mirror_provenance() -> dict[str, Any]:
    """What data snapshot this was trained from."""
    path = config.path("data/mirror/manifest.json")
    if not path.exists():
        return {}
    m = json.loads(path.read_text(encoding="utf-8"))
    return {
        "as_of": m.get("as_of"),
        "max_ingested_at": m.get("max_ingested_at"),
        "row_counts": m.get("row_counts", {}),
        "adjustment_epochs": m.get("adjustment_epochs", []),
    }


@dataclass(frozen=True)
class FrozenModel:
    """A fitted estimator plus everything needed to trust a score it produces."""

    estimator: Any
    manifest: dict[str, Any]
    path: Path

    @property
    def features(self) -> list[str]:
        return list(self.manifest["features"])

    @property
    def model_hash(self) -> str:
        return self.manifest["model_hash"]

    @property
    def trained_through(self) -> str:
        return self.manifest["train_end"]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Score a feature frame, refusing anything that is not what was fitted.

        The check is exact and both ways. A missing column is obvious; an EXTRA
        one is the dangerous case, because dropping it silently would let a
        newly registered feature change nothing and look like the registry and
        the model still agree when they do not.
        """
        want, have = self.features, list(X.columns)
        if set(want) != set(have):
            missing = sorted(set(want) - set(have))
            extra = sorted(set(have) - set(want))
            raise ValueError(
                f"feature mismatch against {self.path.name}: "
                f"missing={missing} extra={extra}. The registry has moved since this model "
                f"was frozen — re-freeze rather than trading a model on features it never saw."
            )
        # Reorder only after the sets are known equal. Order alone is not an
        # error worth refusing over, but it IS worth correcting explicitly:
        # LightGBM reads the matrix positionally and would not complain.
        return self.estimator.predict(X[want])


def build(
    model_name: str = "lightgbm",
    with_cv: bool = False,
    verbose: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Fit on the whole sample and write the artifact + manifest.

    with_cv also runs walk-forward (~7 min) and records the result. Those scores
    describe the PIPELINE, not this artifact — they come from different fits on
    subsets — and the manifest labels them that way.
    """
    cfg = config.load("model")["dataset"]
    horizon = config.load("features")["label"]["horizon"]

    data = ds.load(horizon, include_holdout=True)
    if verbose:
        print(f"  rows                {len(data):,}")
        print(f"  span                {pd.to_datetime(data.dates).min().date()} .. "
              f"{pd.to_datetime(data.dates).max().date()}")
        print(f"  features            {data.X.shape[1]}")

    model = tr.make_model(model_name)
    model.fit(data.X, data.y)

    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_type": model_name,
        "target": cfg.get("target", "residual"),
        "horizon": horizon,
        "features": list(data.X.columns),
        "n_features": data.X.shape[1],
        "train_rows": len(data),
        "train_start": str(pd.to_datetime(data.dates).min().date()),
        "train_end": str(pd.to_datetime(data.dates).max().date()),
        "holdout_included": True,
        "holdout_start": cfg.get("holdout_start"),
        "config_hashes": {
            name: _sha256(config.CONFIG_DIR / f"{name}.yaml")
            for name in ("model", "features", "backtest")
            if (config.CONFIG_DIR / f"{name}.yaml").exists()
        },
        "params": model.get_params(),
        "versions": _versions(),
        "data": _mirror_provenance(),
    }

    if with_cv:
        if verbose:
            print("\n  walk-forward (for the record; a different fit to the one being frozen)")
        run = tr.walk_forward(model_name, horizon=horizon, verbose=verbose)
        manifest["reference_scores"] = {
            "note": (
                "Measured on SEPARATE fits over subsets of this sample, not on the frozen "
                "estimator — a model trained on everything has no out-of-sample left. These "
                "describe the pipeline's edge, not this artifact's."
            ),
            "walk_forward_mean_ic": run.mean_ic,
            "folds": [f.__dict__ for f in run.folds],
        }

    out_dir = config.path(MODEL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The hash covers the manifest, not the pickle: two fits of the same
    # estimator on the same data are not byte-identical (thread scheduling), but
    # the inputs that decide what the model IS are all here.
    payload = json.dumps(manifest, sort_keys=True, default=str)
    model_hash = hashlib.sha256(payload.encode()).hexdigest()[:8]
    manifest["model_hash"] = model_hash

    stem = f"{model_name}_{manifest['train_end']}_{model_hash}"
    pkl, meta = out_dir / f"{stem}.pkl", out_dir / f"{stem}.json"

    import joblib

    joblib.dump(model, pkl, compress=3)
    meta.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")

    if verbose:
        print(f"\n  hash                {model_hash}")
        print(f"  -> {pkl}")
        print(f"  -> {meta}")
    return pkl, manifest


def latest(model_name: str = "lightgbm", model_dir: str | Path | None = None) -> Path:
    """Newest frozen artifact, by the date in its name then by mtime.

    Sorting on the FILENAME date first is deliberate: re-freezing an older
    training window later (say to reproduce a past decision) would have a newer
    mtime, and picking it would quietly trade a stale model.
    """
    d = Path(model_dir) if model_dir else config.path(MODEL_DIR)
    found = sorted(d.glob(f"{model_name}_*.pkl")) if d.exists() else []
    if not found:
        raise FileNotFoundError(
            f"no frozen {model_name} model in {d} — run `vnr freeze` first"
        )
    return max(found, key=lambda p: (p.stem.split("_")[1], p.stat().st_mtime))


def load(path: str | Path | None = None, model_name: str = "lightgbm") -> FrozenModel:
    """Load a frozen model and its manifest. Defaults to the newest."""
    import joblib

    p = Path(path) if path else latest(model_name)
    meta = p.with_suffix(".json")
    if not meta.exists():
        raise FileNotFoundError(
            f"{p.name} has no manifest at {meta.name}. Without it the feature order is "
            f"unknown, and scoring on the wrong order fails silently — refusing to load."
        )
    return FrozenModel(
        estimator=joblib.load(p),
        manifest=json.loads(meta.read_text(encoding="utf-8")),
        path=p,
    )
