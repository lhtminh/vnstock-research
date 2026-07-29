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

LightGBM across 6 purged walk-forward folds: **mean IC +0.0975**, positive in
every fold. Backtest 2013–2026, top 30 equal-weighted, weekly rebalance:

```
Total Return      607.7%     VNINDEX  78.6%
CAGR               15.6%     VNINDEX   8.8%
Max Drawdown       50.4%
Sharpe              0.99
```

Allowing fills on limit-locked bars would report **743.6%** instead. That
136-point gap is the phantom return the tradeability filter removes.

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
