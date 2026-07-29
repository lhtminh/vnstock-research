"""Cross-validation that does not leak.

Plain K-fold is wrong here and the wrongness is invisible: it reports a good
score and the strategy then fails live.

Two reasons. First, a 5-session label at date t is built from prices up to
t+6, so a training row just before the test block already contains the test
block's outcome — that overlap has to be PURGED. Second, adjacent days share
almost the same features and almost the same label, so a random split puts
near-duplicates on both sides and measures memorisation.

Walk-forward with purging fixes both: train only on the past, and drop the
tail of the training set whose labels reach into the test window.

    train ........................|<-purge->|  test  |<-embargo->|
                                  ^
                                  cut here, h+1 sessions before the test starts

The embargo covers the other direction — features are built from trailing
windows, so a row just after the test block still overlaps it. It only matters
when a later fold trains on data following an earlier test block.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PurgedWalkForward:
    """Expanding-window splits over unique dates, with purge and embargo.

    n_splits        number of test blocks
    horizon         label length in sessions; sets the purge width
    embargo         extra sessions dropped after a test block
    min_train_days  a fold with too little history is not worth scoring
    """

    n_splits: int = 5
    horizon: int = 5
    embargo: int = 5
    min_train_days: int = 500

    def split(self, dates: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        d = pd.to_datetime(pd.Series(dates)).to_numpy()
        uniq = np.unique(d)
        if len(uniq) < self.min_train_days + self.n_splits:
            raise ValueError(f"only {len(uniq)} sessions; not enough for {self.n_splits} folds")

        # Test blocks tile the tail of the sample, leaving min_train_days up front.
        first_test = self.min_train_days
        blocks = np.array_split(uniq[first_test:], self.n_splits)

        for block in blocks:
            if len(block) == 0:
                continue
            test_start, test_end = block[0], block[-1]

            # Purge: a training label must have finished before the test begins.
            purge_idx = max(np.searchsorted(uniq, test_start) - (self.horizon + 1), 0)
            train_cutoff = uniq[purge_idx]

            train_mask = d < train_cutoff
            test_mask = (d >= test_start) & (d <= test_end)
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            yield np.flatnonzero(train_mask), np.flatnonzero(test_mask)

    def describe(self, dates: pd.Series) -> pd.DataFrame:
        """Fold boundaries, for eyeballing that the split is what you meant."""
        d = pd.to_datetime(pd.Series(dates))
        rows = []
        for i, (tr, te) in enumerate(self.split(dates)):
            rows.append(
                {
                    "fold": i,
                    "train_start": d.iloc[tr].min().date(),
                    "train_end": d.iloc[tr].max().date(),
                    "test_start": d.iloc[te].min().date(),
                    "test_end": d.iloc[te].max().date(),
                    "train_rows": len(tr),
                    "test_rows": len(te),
                    "gap_days": (d.iloc[te].min() - d.iloc[tr].max()).days,
                }
            )
        return pd.DataFrame(rows)
