# CLAUDE.md

Guidance for any AI agent working here. Short on purpose — the code carries its
own reasoning. This is the map plus what is not obvious from one file.

## What this is

Python 3.13. Reads the `vnstock-service` Postgres database via DuckDB, mirrors
it to Parquet, and does research on top: cleaning, labelling, features, alpha
measurement, LightGBM/XGBoost, vectorbt backtest.

Never modify `D:\vnstock-service`. It is finished, merged and on a scheduler.

## Invariants — do NOT violate these

1. **No feature may look forward.** Every window is `ROWS BETWEEN n PRECEDING
   AND CURRENT ROW`. `registry.win()` is the only way to build a frame and
   cannot express FOLLOWING. `test_features_are_truncation_invariant` is what
   enforces it; if you add a feature and that test fails, the feature is wrong.
2. **Entry is the next session's open, and it must be tradeable.** A signal
   from today's close cannot transact at today's close, and cannot buy a
   limit-locked bar at all.
3. **A missing number is NULL, never 0.** Watch DuckDB specifically:
   `GREATEST(1 - NULL, 0)` returns **0**, not NULL. That silently fabricated
   every pre-2019 `idio_vol_60`. Guard with an explicit `CASE ... IS NOT NULL`.
4. **Rank cross-sectionally within the day's universe**, never against the full
   sample — that would use tomorrow's universe today.
5. **CV must purge.** A 5-session label reaches 6 sessions forward, so training
   rows within that window of a test block share its outcome. Plain K-fold here
   reports a score that cannot be reproduced live.
6. **Only `bar_status = 'normal'` may be filled.**
7. **Ties must break deterministically.** `NTILE` over percentile ranks without
   a tiebreaker returns different bucket means on every run.
8. **Never report total return alone.** Print beta, alpha and annual excess.
   A long-only book in a rising market earns beta x index for free; summing
   that with any real edge makes a tracker look like a strategy.
9. **Weights are a COMPLETE vector on each rebalance date**, zeros included,
   then forward-filled. Blanking zeros before `ffill` makes dropped positions
   immortal — that bug reached 1,156% deployed and produced a fake 607% return.
10. **The holdout is frozen.** Do not train on it, tune against it, or look at
    it more than necessary. It is the only measurement development never saw.

## Things that look like bugs but are not

| Looks wrong | Actually |
|---|---|
| market features NULL before 2019-09-12 | `index_series` starts there; bars start 2002 |
| `suspect_ohlc` = 7,070, not 552 | 552 is vn-audit's *liquid-universe* count; 7,070 is the `volume > 0` total, and matches exactly |
| limit tolerance has a 0.3% floor | stored prices are ADJUSTED so not tick-aligned; a real limit-up can compute to 6.9%. Missing one is worse than over-calling one |
| `band_anomaly` status exists | `symbols.exchange` is the CURRENT venue, so an old bar from a previous listing is judged against the wrong band |
| exit is not required to be tradeable | requiring it would drop limit-up exits, truncating the winning tail and biasing against momentum |
| model IC > best single feature | expected from combining 28 features; the leakage controls are what confirm it |
| the target is residual, not simple excess | simple excess is rank-invariant — 99.8% of rows share their day's benchmark window, so subtracting it changes no ordering. Only the beta term varies per stock |
| holdout IC (0.123) > walk-forward IC (0.083) | not a bug. Recent years genuinely rank better; it also means nothing was overfitted to the dev period |
| strong IC and a losing backtest | the signal is fine and the construction is not: 73.7% turnover x 0.6% round trip = 22.3%/yr against a ~19%/yr gross edge |
| horizon-21 labels are NOT used despite IC decay favouring them | tested. Individual features do strengthen out to 21 sessions, but the combined model's IC FALLS (0.083 -> 0.067) and gets unstable across folds. Lower cost did not make up for it |
| the backtest parameters look under-tuned | deliberate. 35 dev configurations were searched; dev and holdout alpha rank them in opposite orders. More searching fits noise |

## Where the reasoning lives

- `clean/bands.py` — price limits, tick sizes, and why the tolerance is clamped
- `clean/bars.py` — bar_status precedence, generated from the band constants
- `label/forward.py` — entry/exit asymmetry
- `model/cv.py` — the purge diagram
- `backtest/engine.py` — NaN prices as the untradeable mechanism

## Build, run, test

```bash
.venv/Scripts/python -m pytest -q          # 43 tests, no database needed
.venv/Scripts/vnr pipeline                 # needs Postgres on 5432
.venv/Scripts/vnr train --controls         # ~7 min, runs the leakage checks
```

Install core before extras, and always with `-e`: `pip install ".[backtest]"`
without `-e` overwrites the editable install with an empty package.

vectorbt pins the stack tightly (numpy ≥2.4.6, pandas ≥3.0.3) and downgrades
numpy on install. `requirements.lock.txt` holds the combination that was
verified to import together.

## Conventions

- Features are SQL expressions in a registry, not Python callables — one
  definition shared by alpha analysis, training and the backtest.
- Heavy passes (~1M rows) go in DuckDB; summarising a few thousand daily values
  goes in pandas. Do not UNPIVOT wide float tables — `STDDEV_SAMP` overflows.
- Config lives in `config/*.yaml`, not in code.
- `data/` is derived and git-ignored — rebuild it with `vnr pipeline`.
- `reports/` IS committed. Each file is stamped with the data snapshot it came
  from, so the history shows how findings moved as the data filled in.
