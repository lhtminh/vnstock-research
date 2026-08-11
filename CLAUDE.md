# CLAUDE.md

Guidance for any AI agent working here. Short on purpose — the code carries its
own reasoning. This is the map plus what is not obvious from one file.

## What this is

Python 3.13. Reads the `vnstock-service` Postgres database via DuckDB, mirrors
it to Parquet, and does research on top: cleaning, labelling, features, alpha
measurement, LightGBM/XGBoost, vectorbt backtest.

Never modify `D:\vnstock-service` from here. It fetches and stores; this models.

`vnr publish` writes the labelled sample back into that database, but into a
**`research` schema** — `public` is the service's, it applies its own migrations
from Go with unqualified DDL, and nothing here writes to it. The separate schema
is what makes that mechanical instead of a convention.

**This package is also imported as a library by `D:\vnstock-paper`**, which
paper trades the model on live data. That is why three things exist:

- `VNRESEARCH_ROOT` overrides the repo root in `config.py`, so an outside
  importer resolves `config/` and `data/` deliberately rather than by accident.
- `vnr freeze` writes `models/frozen/*.pkl` plus a manifest. Nothing else here
  persists a model — `walk_forward` and `evaluate_holdout` measure a PROCEDURE
  and discard the estimator. Paper trading needs one specific fitted object.
- `backtest/metrics.py` holds `performance` and `annual_table`, and
  `backtest/__init__` resolves submodules lazily. Both exist so that reporting a
  live book does not import vectorbt, which is an optional extra here because it
  pins numpy and pandas hard.

`models/frozen/` is COMMITTED, like `reports/`. It is the record of what was
actually traded on which day.

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
| market features NULL before 2004-01-05 | `index_series` starts there; bars start 2002. (This said 2019-09-12 until the index history was extended. The SAMPLE starts 2009 for a different reason — see `model.yaml`: 2004-2008 has no cross-section to rank) |
| `suspect_ohlc` = 7,070, not 552 | 552 is vn-audit's *liquid-universe* count; 7,070 is the `volume > 0` total, and matches exactly |
| limit tolerance has a 0.3% floor | stored prices are ADJUSTED so not tick-aligned; a real limit-up can compute to 6.9%. Missing one is worse than over-calling one |
| `band_anomaly` status exists | `symbols.exchange` is the CURRENT venue, so an old bar from a previous listing is judged against the wrong band |
| exit is not required to be tradeable | requiring it would drop limit-up exits, truncating the winning tail and biasing against momentum |
| model IC > best single feature | expected from combining 59 features; the leakage controls are what confirm it |
| the target is residual, not simple excess | simple excess is rank-invariant — 99.8% of rows share their day's benchmark window, so subtracting it changes no ordering. Only the beta term varies per stock |
| holdout IC (0.138) > walk-forward IC (0.083) | not a bug. Recent years genuinely rank better; it also means nothing was overfitted to the dev period |
| strong IC and a losing backtest | the signal is fine and the construction is not: 73.7% turnover x 0.6% round trip = 22.3%/yr against a ~19%/yr gross edge |
| horizon-21 labels are NOT used despite IC decay favouring them | tested. Individual features do strengthen out to 21 sessions, but the combined model's IC FALLS (0.083 -> 0.067) and gets unstable across folds. Lower cost did not make up for it |
| the backtest parameters look under-tuned | deliberate. 35 dev configurations were searched; dev and holdout alpha rank them in opposite orders. More searching fits noise |
| `check_single_epoch` allows several epochs in `daily_prices` | it checks PER TICKER, mirroring `v_basis_seams`. Repairing one restatement bumps one ticker, so a healthy database routinely holds two epochs — 12 tickers on 2 and 1,675 on 1 was normal. The old table-wide test failed all of them for the sins of none |
| `vnr freeze` trains on the holdout | that is the point. Going live is what the holdout was kept for, its verdict is already recorded, and the paper log becomes the new out-of-sample test — a better one, because those decisions cannot be recomputed |
| `freeze` reports `train_end` ~6 sessions before the last bar | a 5-session forward label needs 6 sessions ahead to exist. The most recent bars have no label yet |
| a higher IC can come with a WORSE book | the two measure different things and there are now two metrics for that reason. `daily_ic` scores an ordering of ~280 names; `topn_edge` scores the 50 that get bought. Judge a change on the second. The first led to `exit_rank: 200`, which cost 80% of the edge before anything measured it |
| the buffer is not free | `exit_rank` was set to cut turnover and it did — 73.7% -> 8.9%. What nobody measured was the signal given up: at 200 the daily top-50 edge is +0.00350 and what the book actually HOLDS earns +0.00068. Moving to 100 took dev alpha -5.18% -> +1.21% with drawdown unchanged. See `config/backtest.yaml` for the sweep |
| 59 features rank 15% better and earn LESS on dev | measured, not a bug in either number. WF IC +0.0730 -> +0.0838; dev alpha -2.93% -> -3.77%. 8 of 14 years improved and the loss is almost entirely 2016 (-8.3pp) and 2017 (-10.5pp); 2020-2023 gained +10.6pp between them. Both configurations are NEGATIVE on dev, so this compares two losing setups over a period where the long-only book has never worked. It is the third piece of evidence that portfolio construction, not prediction, is the binding constraint |
| peer groups got smaller (7.7 -> 5.9 names, corr 0.388 -> 0.357) | expected, and it is the fix working. Demeaning by the INVESTABLE cross-section removes more of the common factor than demeaning by every ticker in the panel did, so less residual correlation is left and fewer pairs clear `MIN_CORR`. That threshold was calibrated against the old, inflated numbers — it is now effectively stricter. Not retuned, because tuning it against the same data that revealed it is how the dev/holdout orderings got reversed before |
| only 58 rows in 2.5M are `dau_co_manh` | the mentor's weighting, and **do not "fix" it**. PVDI carries 0.40 on ABSOLUTE thresholds (1/2/3.5) and reaches its top bucket on 0.03% of rows, while the other three are percentile-scored and fill theirs by construction. Rescoring PVDI on percentiles makes the bucket reachable and makes the labels WORSE at predicting forward returns — worse in 8 of 8 pairwise comparisons, the most damaging of three candidate changes. Rare by design is the point. Tested in `reports/speculation-labelling-20260810.md` |
| `spec_label_adj` differs from `spec_label` in only ONE setting | it used to be three. Volatility→magnitude helped (8 of 8), range→pct did nothing consistent, PVDI→percentile hurt. Both were reverted on measurement. A test pins the difference at exactly one setting so the reverts are not undone by someone who remembers the argument but not the result |
| `research.features` stores REAL, but the parquet is DOUBLE | deliberate. `dataset.load()` casts every feature to float32 before the model sees one, so the extra bits are never consumed — 505 MB instead of 918 MB, and both paths round the same double to the same float32. Prices, returns and labels stay DOUBLE, because that reasoning does not cover them |
| a feature with a NULL raw value has a NULL rank, not 0.5 | correct, and it was a real defect until `c2d20a9`. `PERCENT_RANK() OVER (ORDER BY col)` sorts NULLs last, so every row missing a value was bunched at the TOP of that day's ranking: measured 2026-08-04, 47 names with no `peer_corr` all scored 0.836 while the 234 real values spanned 0.000-0.832 — "no data" read to the model as "higher than 83% of the market". Invariant 3 one layer up. Fixed by ranking only real values; imputing a middle value instead would be inventing data. Verified on the published table: 317,121 rows have no `peer_corr` and **none** carries a rank |

## Where the reasoning lives

- `clean/bands.py` — price limits, tick sizes, and why the tolerance is clamped
- `features/technical.py` — why RSI and MACD are simple, not exponential
- `features/shape.py` — the two ways to measure speculation, one of them Vietnam-only
- `clean/bars.py` — bar_status precedence, generated from the band constants
- `label/forward.py` — entry/exit asymmetry
- `label/speculation.py` — the mentor's scheme, and the three places it and the
  data disagree. Not a training target: every input is trailing
- `model/cv.py` — the purge diagram
- `backtest/engine.py` — NaN prices as the untradeable mechanism
- `io/publish.py` — the schema boundary, and why the holdout is its own view

## Build, run, test

```bash
.venv/Scripts/python -m pytest -q          # 124 tests, no database needed
.venv/Scripts/vnr pipeline                 # needs Postgres on 5432
.venv/Scripts/vnr train --controls         # ~7 min, runs the leakage checks
.venv/Scripts/vnr freeze                   # ~10 min, writes models/frozen/
.venv/Scripts/vnr publish                  # ~20s, labelled sample -> research schema
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
