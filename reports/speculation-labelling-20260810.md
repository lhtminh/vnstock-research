# Speculation labelling — both label sets, and three things to take back to the mentor

> **Update, same day.** Both variants are now built on every run and sit side by
> side in the same table: `spec_label` is the document exactly as written,
> `spec_label_adj` applies the three corrections measured below. Nothing else
> differs — same four dimensions, same 40/30/15/15 weights, same 0-3 scoring,
> same 0.75/1.5/2.25 thresholds — and `test_speculation.py` enforces that the
> gap stays exactly three settings wide.
>
> | | Bình thường | Đầu cơ nhẹ | Đầu cơ | Đầu cơ mạnh |
> |---|---|---|---|---|
> | **mentor** | 83.87% | 15.44% | 0.69% | **0.002%** (58 rows) |
> | **adjusted** | 77.73% | 19.58% | 2.62% | **0.07%** (1,721 rows) |
>
> They disagree on **381,397 rows**, 15.0% of the 2,544,847 that carry a score.
> The top bucket goes from unreachable to rare-but-real, which is what it was
> presumably meant to be. Component distributions after the change:
>
> | component | Bình thường | nhẹ | Đầu cơ | mạnh |
> |---|---|---|---|---|
> | PVDI (mentor, fixed cut-offs) | 80.02% | 16.55% | 3.41% | **0.03%** |
> | PVDI (adjusted, percentiles) | 74.81% | 15.06% | 5.07% | **5.05%** |
> | Turnover — shared, one form only | 75.04% | 14.55% | 5.05% | 5.37% |
> | Volatility (signed → magnitude) | 75.41% | 14.79% | 4.90% | 4.91% |
> | | 75.36% | 14.70% | 4.92% | 5.01% |
> | Range (VND → share of price) | 73.71% | 15.40% | 5.29% | 5.60% |
> | | 75.01% | 14.84% | 4.95% | 5.19% |
>
> Note what this shows about volatility and range: the *distributions* barely
> move, because percentile scoring fills the buckets by construction either way.
> It is **which stocks** land in them that changes — 35.4% and 33.0% of daily
> labels respectively. A distribution table cannot show a broken measure; only
> the disagreement counts and the crash test below can.


Source: `Phương Pháp Nhận Diện Dấu Hiệu Đầu Cơ Cổ Phiếu`
Built 2026-08-10 from `data/clean/panel.parquet`, mirror `2026-08-10T08:59:25Z`.
Code `label/speculation.py`, settings `config/speculation.yaml`,
published to `research.speculation`.

## What was built

All four components, the document's weights, the document's thresholds.

```
Score = 0.40·PVDI + 0.30·Turnover + 0.15·Volatility + 0.15·Range
```

**2,922,876 rows, 2002-04-19 to 2026-08-10.** 2,544,847 carry a composite score;
the other 378,029 are warm-up — a stock needs 252 sessions of its own history
before PVDI means anything, and those rows are NULL rather than guessed.

| component | Bình thường | Đầu cơ nhẹ | Đầu cơ | Đầu cơ mạnh |
|---|---|---|---|---|
| PVDI | 80.02% | 16.55% | 3.41% | 0.03% |
| Turnover | 75.04% | 14.55% | 5.05% | 5.37% |
| Volatility | 75.41% | 14.79% | 4.90% | 4.91% |
| Range | 73.71% | 15.40% | 5.29% | 5.60% |
| **Composite** | **83.87%** | **15.44%** | **0.69%** | **0.002%** |

The three percentile components land on 75/15/5/5 exactly as designed. PVDI does
not, and that turns out to matter — see finding 3.

## 1. Turnover cannot use free float, because there is none

The document divides period volume by average free float. This database has no
free-float history and no share-count history to derive one from:

| source | coverage |
|---|---|
| `company_snapshot.issue_share` | **51 tickers, 1 date** |
| `market_snapshot.listed_share` | 1,620 tickers, but **4 real sessions**, all within the last two weeks |

Applying a 2026 share count to a 2010 bar would be wrong twice over — the count
is different, and it is information from the future. So the denominator is the
stock's **own trailing 252-session average volume**, against a 21-session
numerator. That keeps what the measure is for (activity abnormal relative to
this name's normal) and keeps it dimensionless, which is what the
cross-sectional percentile step requires.

**This is the one departure that changes the measure itself.** Free float is a
better denominator: it distinguishes a stock whose whole float turns over
weekly from one merely busier than usual. Switch `turnover.denominator` to
`free_float` once the service has the history — the service ingests
`company_snapshot` weekly, so this improves on its own.

## 2. The Volatility formula cannot see a crash

The formula is the **signed** daily change averaged over 5 sessions. The theory
section two paragraphs above it says volatility rises both when a stock is
pushed up *and* when it is dumped — `giảm đột ngột do bán tháo`. A signed mean
cannot do the second: a falling stock sits at the **bottom** of the
distribution, and the buckets only flag the top.

Measured on the full panel, signed against magnitude:

- **35.4%** of daily labels differ (1,032,484 of 2,916,336)
- **14,561** rows where a stock fell more than 5% a day averaged over five
  sessions. The signed formula labels **all 14,561** `Bình thường`.
- 78,857 rows it calls `Bình thường` are `Đầu cơ mạnh` under magnitude

100% of the crash cases are missed. `volatility.measure: abs` is a one-word
change and needs the mentor's confirmation, not mine — the formula is
unambiguous even though it contradicts the paragraph above it.

## 3. Range in absolute VND mostly ranks price

`Range = P_high − P_low`, unnormalised, compared across the whole market. A
100,000 VND stock has a wider absolute range than a 5,000 VND one on the same
percentage move, so the label tracks price level.

Median share price by label:

| | Bình thường | Đầu cơ nhẹ | Đầu cơ | Đầu cơ mạnh |
|---|---|---|---|---|
| absolute (the document) | 6,910 | 15,000 | 21,740 | **34,780** |
| ÷ previous close | 9,360 | 7,650 | 6,950 | **6,500** |

Monotone increasing under the document's formula: the "most speculative" bucket
is the expensive half of the market. Dividing by the previous close **inverts**
it, which is the direction penny-stock speculation actually runs in Vietnam. The
two disagree on 33.0% of daily labels.

## 4. The top composite bucket is nearly unreachable

**58 rows of 2,544,847 — 0.002%** are `Đầu cơ mạnh`.

Not a bug, an interaction. PVDI carries the largest weight (0.40) but is scored
against **absolute** thresholds (1 / 2 / 3.5) while the other three are scored
against **percentiles**, which guarantee 5% in each top bucket by construction.
PVDI reaches its top bucket on 0.03% of rows. Reaching 2.25 needs PVDI at 3 —
0.4·3 = 1.2 — plus almost everything else at maximum.

Two coherent fixes, both the mentor's call:

- score PVDI by percentile too, so all four components are on one scale; or
- lower the composite thresholds, since 0.75/1.5/2.25 were presumably chosen
  assuming four comparable components.

## Reading it

```sql
SELECT ticker, spec_score, spec_label, spec_score_adj, spec_label_adj
FROM research.speculation
WHERE date = '2026-08-10' AND spec_label_adj <> 'binh_thuong'
ORDER BY spec_score_adj DESC;
```

Where the two schemes disagree, which is the argument worth having:

```sql
SELECT spec_label, spec_label_adj, count(*)
FROM research.speculation
WHERE spec_label IS NOT NULL AND spec_label <> spec_label_adj
GROUP BY 1, 2 ORDER BY 3 DESC;
```

Component labels are `pvdi_label`, `turnover_label`, `vol_label`, `range_label`,
each with an `_adj` twin except turnover, which has only one form here. The
document's 12-month "nhãn tổng thể" are `vol_label_12m` and `range_label_12m`,
also with `_adj` twins.

## What this is not

It is **not** a training target. Every input is trailing, so it describes how a
stock is behaving now rather than what it will do next. `label/forward.py`
remains the model's label. This is a feature, a filter, or a report — training
on it would teach a model to recognise speculation that is already visible.

## Two defects found while building

Both were in this code, not the document, and both are the same shape as bugs
this repo has hit before.

- **`CORR` returns NaN, not NULL**, when a window has no variance — 2,874 rows,
  every stock that traded flat volume for a whole window. NaN survives every
  `IS NULL` guard and made `STDDEV_SAMP` raise outright. Invariant 3 in a new
  disguise; `finite()` converts at the source.
- **`ROWS BETWEEN 251 PRECEDING` does not require 252 rows.** It computes over
  whatever exists, so a stock's third session was getting a "12-month
  correlation" from three points — PVDI was non-NULL on 99.85% of rows when a
  252-session measure cannot be. `features/build.py` NULLs its warm-up for
  exactly this reason.
