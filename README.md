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

That runs mirror → clean → panel → label → features. Then:

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
| `features` | `data/features/features.parquet` | 28 features plus cross-sectional ranks |
| `alpha` | `reports/alpha-*.md` | IC, decay, quantile spread, turnover, redundancy |
| `train` | `data/oos_*.parquet` | Walk-forward with purged folds |
| `backtest` | stdout | vectorbt, VN costs, capacity cap |

Everything is DuckDB SQL over Parquet. The full pipeline is about three minutes
on 4.8M bars.

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
- **Market features start 2019-09-12**, where `index_series` starts. Bars go
  back to 2002. Beta, alpha and excess return are NULL before that — a boundary,
  not a bug.

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

LightGBM on the residual target, 4 purged walk-forward folds (2021–2023):
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

43 tests. The ones that matter:

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
  features/   registry + momentum/price/volume/market/xsec
  alpha/      IC, quantiles, report
  model/      purged CV, dataset, walk-forward training
  backtest/   vectorbt engine, VN cost model
```
