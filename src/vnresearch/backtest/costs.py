"""Vietnamese transaction costs.

Round-trip is well over 0.5%, which is the number that kills most daily
signals. Anything rebalancing more than weekly needs a spread large enough to
pay this twice a week, and almost nothing has one.
"""

from __future__ import annotations

from dataclasses import dataclass

from vnresearch import config


@dataclass(frozen=True)
class Costs:
    brokerage: float  # per side
    sell_tax: float  # sales only
    slippage: float  # per side

    @classmethod
    def load(cls) -> Costs:
        c = config.load("backtest")["costs"]
        return cls(
            brokerage=float(c["brokerage"]),
            sell_tax=float(c["sell_tax"]),
            slippage=float(c["slippage"]),
        )

    @property
    def symmetric_fee(self) -> float:
        """One per-side fee for vectorbt, with the sell tax split across both.

        The tax really lands only on sales, but a strategy that buys must
        eventually sell, so charging half each way costs the same over a round
        trip and keeps the model to a single number.
        """
        return self.brokerage + self.sell_tax / 2

    @property
    def round_trip(self) -> float:
        return 2 * self.brokerage + self.sell_tax + 2 * self.slippage

    def breakeven_spread(self, turnover: float) -> float:
        """Return per period a signal must earn just to cover its trading."""
        return turnover * self.round_trip
