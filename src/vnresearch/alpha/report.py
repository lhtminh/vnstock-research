"""Markdown summary of a feature study, stamped with the data snapshot.

The stamp is not decoration. Feature results move when the data underneath
moves — a corporate action restates prices, a weekly job fills in sectors — and
without the manifest id you cannot tell that from a change you made.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from vnresearch import config
from vnresearch.alpha import ic as ic_mod
from vnresearch.alpha import quantiles
from vnresearch.io.manifest import load_manifest

# Above this on daily data, suspect the pipeline before believing the signal.
LEAKAGE_SUSPECT_IC = 0.15


def _fmt(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".4f")


def build(horizon: int = 5, horizons: list[int] | None = None, top: int = 10) -> Path:
    horizons = horizons or config.load("features")["label"]["horizons"]
    m = load_manifest(config.path(config.load("data")["mirror_dir"]))

    table = ic_mod.ic_table(horizon)
    decay = ic_mod.ic_decay(horizons)
    strongest = table.head(top)["feature"].tolist()

    spreads = pd.DataFrame([quantiles.spread_summary(f, horizon) for f in strongest])
    spreads["turnover"] = [quantiles.turnover(f, step=horizon) for f in strongest]

    corr = ic_mod.correlation(strongest)
    pairs = [
        (a, b, corr.loc[a, b])
        for i, a in enumerate(strongest)
        for b in strongest[i + 1 :]
        if abs(corr.loc[a, b]) > 0.85
    ]

    suspect = table[table["mean_ic"].abs() > LEAKAGE_SUSPECT_IC]["feature"].tolist()

    out = config.path("reports")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"alpha-{m.id}-h{horizon}.md"

    lines = [
        f"# Alpha report — {horizon}-session horizon",
        "",
        f"- data snapshot: `{m.as_of}` (epochs {m.adjustment_epochs})",
        f"- generated: {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"- mirrored rows: {sum(m.row_counts.values()):,}",
        "",
        "## Information coefficient",
        "",
        "Spearman correlation of each feature's daily cross-sectional rank with the",
        "forward-return rank. `ir` = mean/sd across days; it matters more than mean IC.",
        "",
        _fmt(table.head(top * 2)),
        "",
        "## IC decay by horizon",
        "",
        "Where IC fades is the natural holding period. A signal that peaks at 21",
        "sessions should not be traded weekly.",
        "",
        _fmt(decay.set_index("feature").loc[strongest].reset_index()),
        "",
        "## Quantile spread and turnover",
        "",
        "`spread_pct` is top bucket minus bottom, per period. `turnover` is the",
        "fraction of the top bucket replaced each rebalance — multiply by round-trip",
        "cost (>0.5% in Vietnam) to see what the spread has to beat.",
        "",
        _fmt(spreads),
        "",
    ]

    if pairs:
        lines += [
            "## Redundant pairs (|corr| > 0.85)",
            "",
            "These carry the same information. Feeding both to a tree splits their",
            "importance and makes each look unimportant.",
            "",
            _fmt(pd.DataFrame(pairs, columns=["a", "b", "corr"])),
            "",
        ]

    if suspect:
        lines += [
            "## Leakage suspects",
            "",
            f"Mean |IC| above {LEAKAGE_SUSPECT_IC} on daily data is rare and usually means",
            "the feature saw the future. Check these before believing them:",
            "",
            *(f"- `{f}`" for f in suspect),
            "",
        ]
    else:
        lines += [
            "## Leakage check",
            "",
            f"No feature exceeds |IC| {LEAKAGE_SUSPECT_IC}. That is the expected result.",
            "",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
