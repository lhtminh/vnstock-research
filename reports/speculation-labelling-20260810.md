# Speculation labelling — the document is better than the critique of it

> **VALIDATION, same day, and it overturns most of what is written below.**
>
> Both schemes were scored against forward returns on the dev period only
> (2009-01-01 .. 2023-12-31 — the holdout was not touched). Outcome is the
> forward return minus that day's cross-sectional mean, which removes the market
> move exactly with no beta estimate to get wrong.
>
> **The labels do predict underperformance, monotonically, in every cell:**
>
> | universe, h=21 | mean excess | forward sd | share losing >20% |
> |---|---|---|---|
> | Bình thường | **+0.15%** | 12.63% | 2.32% |
> | Đầu cơ nhẹ | −0.47% | 14.56% | 4.90% |
> | Đầu cơ | **−1.74%** | 20.19% | **11.51%** |
>
> A name flagged Đầu cơ is **5x more likely to lose a fifth of its value** over
> the next month than an unflagged one. That is the useful finding, and it is
> the document's, not mine.
>
> **Of my three proposed corrections, one helped and one actively hurt.** All
> eight combinations were scored; each change is judged by its four pairwise
> comparisons with the other two settings held fixed, at two horizons — eight
> independent readings each, not a single best-of-eight pick.
>
> | change | verdict | evidence |
> |---|---|---|
> | volatility, signed → magnitude | **helps** | better in 8 of 8 |
> | range, VND → share of price | **no reliable effect** | wins at h=21, loses at h=5, small either way |
> | PVDI, fixed → percentile | **hurts, worst of the three** | worse in 8 of 8 |
>
> Top 2% of each day, universe, mean excess return:
>
> | pvdi | vol | range | h=21 | h=5 |
> |---|---|---|---|---|
> | mentor | **magnitude** | mentor | **−1.72%** | −0.71% |
> | mentor | mentor | mentor — *the document* | −1.11% | −0.42% |
> | percentile | magnitude | pct — *what I proposed* | −0.76% | −0.39% |
> | percentile | mentor | pct | −0.04% | −0.14% |
>
> **So: change exactly one thing.** `adjusted` now differs from `mentor` only in
> the volatility measure; the other two were reverted on evidence.
>
> Why the PVDI change was wrong: rare-by-design *is* the point. Forcing PVDI to
> fire on 5% of rows turns a precise signal into a generic one. The 58-row top
> bucket is a real consequence of the document's weighting — but it is a
> cosmetic complaint, and the fix for it cost more than the complaint was worth.
>
> The range criticism was descriptively true and predictively irrelevant.
> Absolute VND really does track price level; that just does not make it worse
> at this job.
>
> Caveat: eight configurations on dev is a search, and this repo has been burned
> before by dev and holdout ranking configurations in opposite orders. The
> pairwise consistency (8 of 8, across two horizons) is what makes the volatility
> result worth acting on; a best-of-eight pick alone would not be.


> **What shipped.** Both label sets are built on every run and sit side by side:
> `spec_label` is the document exactly as written, `spec_label_adj` changes the
> volatility measure and nothing else. `test_speculation.py` pins the gap at
> exactly one setting, so the two reverts above are not undone later by someone
> who remembers the argument but not the measurement.
>
> | | Bình thường | Đầu cơ nhẹ | Đầu cơ | Đầu cơ mạnh |
> |---|---|---|---|---|
> | **mentor** | 83.87% | 15.44% | 0.69% | 0.002% (58 rows) |
> | **adjusted** | 84.06% | 15.16% | 0.78% | 0.005% (119 rows) |
>
> They disagree on **120,319 rows**, 4.7% of the 2,544,847 that carry a score.
>
> The distributions are nearly identical, and that is the point worth
> understanding: percentile scoring fills the buckets by construction whatever
> you feed it, so the *shape* cannot tell you whether a measure is any good. What
> changes is **which stocks** land in each bucket — 35.4% of daily volatility
> labels move. A distribution table cannot show a broken measure; only the
> disagreement count, the crash test, and the forward-return validation can.
>
> For the record, the earlier three-change variant produced a much flatter
> distribution (77.73 / 19.58 / 2.62 / 0.07%, 1,721 rows in the top bucket) and
> looked more useful for it. It ranked worse. That is why the distribution table
> is not the test.


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
