# vnstock-research

Alpha research on the Vietnamese market: clean → label → features → alpha →
gradient-boosted trees → backtest.

Reads the Postgres database built by `vnstock-service` through DuckDB. That
service fetches and stores; this one has all the opinions about features and
models. Neither repo imports the other.

## Quickstart

```bash
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,backtest]"
```

Install the core first and `.[backtest]` second — installing an extra without
`-e` replaces the editable install and the package silently stops importing.

With Postgres running (`vnstock-pg` container, port 5432):

```bash
.venv/Scripts/vnr pipeline
```

That runs mirror → clean → panel → label → peers → features. Then:

```bash
.venv/Scripts/vnr alpha
```

```bash
.venv/Scripts/vnr train --controls
```

```bash
.venv/Scripts/vnr backtest --compare
```

## Stages

| Command | Output | What it does |
|---|---|---|
| `mirror` | `data/mirror/*.parquet` | Copies 8 Postgres tables locally, with a manifest |
| `clean` | `data/clean/bars.parquet` | Rebuilds `bar_status`, which the source leaves NULL |
| `panel` | `data/clean/panel.parquet` | Tradeability, ADTV, universe, market return |
| `label` | `data/clean/labels.parquet` | Forward returns at horizons 1/5/10/21 |
| `peers` | `data/features/peers.parquet` | Correlation-based peer sets the `peer_*` features read |
| `features` | `data/features/features.parquet` | 59 features plus cross-sectional ranks |
| `alpha` | `reports/alpha-*.md` | IC, decay, quantile spread, turnover, redundancy |
| `train` | `data/oos_*.parquet` | Walk-forward with purged folds |
| `backtest` | stdout | vectorbt, VN costs, capacity cap |
| `speculation` | `data/clean/speculation.parquet` | The mentor's PVDI/turnover/volatility/range labels |
| `publish` | Postgres `research` schema | The labelled sample, for SQL from outside this repo |

Everything is DuckDB SQL over Parquet. The full pipeline is about three minutes
on 4.8M bars.

## Reading the labelled data from SQL

`vnr publish` copies the labelled sample into the **`research` schema** of the
same database the service writes to. `public` is the service's and is never
touched.

| Object | Rows | What |
|---|---|---|
| `research.features` | 943k | The feature matrix, raw + rank, universe rows |
| `research.labels` | 943k | Forward returns at 1/5/10/21, tradeable entry |
| `research.training_sample` | 644k | **view** — the matrix the model trains on |
| `research.holdout_sample` | 212k | **view** — the frozen holdout, kept separate on purpose |
| `research.speculation` | 2.9M | Speculation labels, **both** the mentor's scheme and an adjusted one |
| `research.ratings` | 943k | Dimension Z-scores, composite, speculation penalty, A-E grade |
| `research.dim_*` | — | **views** — one per dimension, raw + rank, nine of them |
| `research.feature_catalog` | 71 | Each feature's dimension, direction, lookback and SQL definition |
| `research.publish_runs` | — | One row per publish, with the mirror snapshot it came from |

The catalog is what makes the matrix readable — otherwise it is 127 unlabelled
float columns, and which of the 8 dimensions a column belongs to lives only in
`Feature.category`, which is Python.

```sql
SELECT dimension, count(*), string_agg(feature, ', ' ORDER BY feature)
FROM research.feature_catalog GROUP BY 1 ORDER BY 2 DESC;
```

| dimension | n | dimension | n |
|---|---|---|---|
| technical | 16 | volatility | 9 |
| liquidity | 11 | beta | 8 |
| momentum | 9 | range | 6 |
| speculation | 6 | peer | 4 |
| season | 2 | | |

Each also has its own view — `research.dim_technical`, `research.dim_volatility`
and so on — carrying that dimension's raw values beside their ranks. The wide
`research.features` table is right for a model and wrong for a person.

`definition` carries the SQL each feature is computed from, which is what makes
a column answerable rather than merely present: `rsi_14` here is a 14-session
**simple** average, not the exponential one a charting package draws, and
reading the expression is the only way to know which you have.

## Ratings

`vnr rating` scores every name in the day's universe: one Z-score per dimension,
a weighted composite, a speculation penalty, and an A-E grade.

```sql
SELECT ticker, rating, rating_score, z_momentum, z_volatility, z_liquidity, spec_label
FROM research.ratings WHERE date = '2026-08-10' ORDER BY rating_score DESC LIMIT 20;
```

It **explains**; the model in `model/` **predicts**. Every dimension keeps its
own column because "strong momentum, thin liquidity" and the reverse are
different positions even at the same overall score, and a portfolio manager
needs to see which one they are holding.

Nothing in it is fitted. Weights are equal by default, and each feature's
direction is DECLARED in the registry from what it measures — a sign fitted to
the sample flips between periods and takes the explanation with it.
`vnr rating --check` reports where the data disagrees (58 of 63 agree today); it
reports and does not flip.

Grades are 10/20/40/20/10 of each day's universe, so a grade is always relative
to what else was tradeable that day.

## Speculation labels

`vnr speculation` implements the mentor's four-component scheme. Both label sets
are built every run, side by side:

| columns | what |
|---|---|
| `spec_score`, `spec_label` | the document exactly as written |
| `spec_score_adj`, `spec_label_adj` | three measured corrections, below |
| `pvdi_label`, `turnover_label`, `vol_label`, `range_label` | components, mentor's |
| `*_label_adj` | components, adjusted (turnover is shared — it has only one form here) |
| `vol_label_12m`, `range_label_12m` (+ `_adj`) | the document's 12-month "nhãn tổng thể" |

The adjusted variant changes **one** thing: volatility from the signed 5-day
mean to its magnitude, because the signed form cannot see a stock falling.
Everything else is the document's.

Two further changes were proposed and then **reverted on measurement** — scoring
PVDI by percentile made the labels worse at predicting forward returns in 8 of 8
pairwise comparisons, and normalising Range by price did nothing consistent.
The document was closer to right than the critique of it.

The labels do carry information. Over the dev period, a name flagged `Đầu cơ`
was **5x more likely to lose more than 20%** in the following month than an
unflagged one (11.5% vs 2.3%), with mean excess return −1.74% against +0.15%.
See `reports/speculation-labelling-20260810.md`.

**Not a training target** — every input is trailing, so it describes the present
rather than predicting the future. `label/forward.py` remains the model's label.

`training_sample` is generated from `config/model.yaml` and the same target
expression `dataset.load()` uses, so it is the training matrix rather than
something shaped like it — `test_publish.py` pins that, and the two were checked
row for row on all 643,548 rows and 59 features.

```sql
SELECT ticker, date, y, target FROM research.training_sample
WHERE date = '2023-12-29' ORDER BY y DESC LIMIT 10;
```

It is **not** part of `vnr pipeline`. The pipeline runs against the clock before
the 15:00 trading decision and nothing on that path reads these tables, so
publishing is a separate ~20s step; `publish_runs.mirror_as_of` against
`data/mirror/manifest.json` tells you whether the copy is current.

## What the data actually holds

The source schema promises more than it currently delivers, so a few things are
thin on purpose:

- **`bar_status` is NULL on all 4.77M source rows**, as are `value`,
  `ref_price`, `ceiling_price`, `floor_price`. `clean/bands.py` reconstructs the
  price limits; turnover is a `close * volume` proxy.
- **Foreign flow is one usable day.** It accumulates ~1,500 rows a session going
  forward. Not modelable yet.
- **Sector covers 37 tickers, market cap 51.** Both grow when the service's
  weekly job runs.
- **Market features start 2004-01-05**, where `index_series` starts. Bars go
  back to 2002. Beta, alpha and excess return are NULL before that — a boundary,
  not a bug. The training sample starts later still, at 2009, for an unrelated
  reason: 2004-2008 has too few liquid names to rank.

Universe: ~293 names on a given recent day, 941k rows across 2002–2026.

## Results, current snapshot

Single features, 5-session horizon — all consistent with the literature:

| Feature | mean IC | Note |
|---|---|---|
| `hl_range` | −0.071 | low-volatility anomaly, biggest spread, 62% turnover |
| `idio_vol_60` | −0.070 | idiosyncratic volatility puzzle; strengthens to −0.103 at 21d |
| `dist_52w_high` | +0.039 | 52-week-high momentum; monotonic, only 24% turnover |
| `ret_1` | −0.034 | short-term reversal, decays to zero by 21d |
| `mom_12_1` | +0.017 | classic momentum, not monotonic here |

LightGBM on the residual target, 4 purged walk-forward folds (2010–2023):
**mean IC +0.0825**, positive in every fold. On the **frozen holdout**
(2024-01-01 onward, never trained on, scored once): **IC +0.1231, IR 1.32**.
The holdout scoring higher than development is the opposite of overfitting.

### The signal works. The portfolio does not.

Holdout, per 5-session period:

```
top decile   raw forward return   +0.380%
bottom       raw forward return   -1.077%
long-short spread                 +1.456%   <- a large, real edge
```

Yet the long-only top-30 weekly book **loses money**:

```
CAGR              -2.59%     VNINDEX  +18.84%
beta                0.82
ALPHA             -18.05%
2024 excess       -12.2%
2025 excess       -30.7%
2026 excess       -12.5%
```

The arithmetic is not subtle:

```
realized turnover per rebalance    73.7%
cost per rebalance                 0.442%
top-decile gross edge per period   0.380%
net                               -0.062%   x50 rebalances a year
```

**Weekly rebalancing of a top-30 book spends 22.3% a year to harvest an edge
worth about 19%.** The next step is portfolio construction — longer holds, lower
turnover, or a long-short book that captures the 1.456% spread instead of the
0.380% long leg — not more features.

Two earlier numbers in this file were wrong and are worth recording. A "607%
total return" came from a weight bug that let dropped positions carry their old
weight forever, so the book grew to 1,156% deployed. And the ~15.6% CAGR it
implied was mostly beta, which is why the decomposition is now printed by
default.

## Verification

```bash
.venv/Scripts/python -m pytest -q
```

93 tests. The ones that matter:

- **truncation invariance** — every feature is recomputed on truncated history
  and must be identical, which is what catches lookahead
- **purge width** — the CV gap must always exceed the label horizon
- **label alignment** — entry is the next session's open, never today's close
- **leakage controls** (`vnr train --controls`) — shuffled labels must give
  IC ≈ 0, and the answer handed over as a feature must give IC ≈ 1

## Layout

```
config/       data, features, model, backtest settings
src/vnresearch/
  io/         DuckDB session, Postgres mirror, manifest
  clean/      price bands, bar_status, panel, calendar
  label/      forward returns
  features/   registry + momentum/price/volume/market/technical/shape/peers/xsec
  alpha/      IC, quantiles, report
  model/      purged CV, dataset, walk-forward training
  backtest/   vectorbt engine, VN cost model
```
