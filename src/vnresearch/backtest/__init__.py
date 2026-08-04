"""Backtest package.

Submodules are resolved LAZILY. `engine` imports vectorbt, which pins numpy and
pandas hard and is an optional extra here for exactly that reason — so importing
it eagerly made `from vnresearch.backtest.costs import Costs` fail without it.
Costs, rebalancing and the risk overlay are plain pandas and have no business
requiring a backtesting library.

`from vnresearch.backtest import engine` still works and still raises if
vectorbt is genuinely missing; it just no longer does so for callers who only
wanted the cost model.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["costs", "engine", "hedge", "rebalance", "risk"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
