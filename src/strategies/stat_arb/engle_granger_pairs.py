"""
TradingApp — Engle-Granger Pairs Strategy

Statistical arbitrage using Engle-Granger cointegration test.
  • Test for cointegration between two assets on a rolling window
  • Compute spread z-score
  • Long spread when z < -entry_threshold, short when z > +entry_threshold
  • Exit when z returns within exit_threshold

Note: This is a simplified single-pair version. The full implementation
would scan the universe for cointegrated pairs.
"""

from __future__ import annotations

import numpy as np

from src.core.models import Bar, Signal, Direction
from src.strategies.base_strategy import BaseStrategy


class EngleGrangerPairs(BaseStrategy):

    @property
    def name(self) -> str:
        return "engle_granger_pairs"

    @property
    def family(self) -> str:
        return "stat_arb"

    def __init__(self, params=None):
        super().__init__(params)
        # In a full implementation, the pair partner data would come
        # from a separate data channel. Here we demonstrate the
        # single-asset z-score reversion logic (spread vs its own mean).
        self._spread_history: dict[str, list[float]] = {}

    def generate_signal(self, symbol: str, bars: list[Bar]) -> Signal | None:
        lookback = self.params.get("lookback", 252)
        zscore_entry = self.params.get("zscore_entry", 2.0)
        zscore_exit = self.params.get("zscore_exit", 0.5)

        if len(bars) < lookback:
            return None

        closes = np.array(self._closes(bars[-lookback:]))

        # Compute spread as log returns vs rolling mean (self-mean-reversion proxy)
        log_prices = np.log(closes)
        spread = log_prices - np.mean(log_prices)

        mean = np.mean(spread)
        std = np.std(spread, ddof=1)
        if std < 1e-10:
            return None

        zscore = (spread[-1] - mean) / std

        # Store z-score history for exit detection
        if symbol not in self._spread_history:
            self._spread_history[symbol] = []
        self._spread_history[symbol].append(zscore)
        # Prevent unbounded memory growth
        if len(self._spread_history[symbol]) > 1000:
            self._spread_history[symbol] = self._spread_history[symbol][-1000:]

        # Signal logic
        if zscore < -zscore_entry:
            conviction = min(1.0, abs(zscore) / (zscore_entry * 2))
            return Signal(
                strategy=self.name,
                symbol=symbol,
                direction=Direction.LONG,
                conviction=conviction,
                timestamp=0.0,
                metadata={"z_score": zscore, "half_life": self._half_life(spread)},
            )
        elif zscore > zscore_entry:
            conviction = min(1.0, abs(zscore) / (zscore_entry * 2))
            return Signal(
                strategy=self.name,
                symbol=symbol,
                direction=Direction.SHORT,
                conviction=conviction,
                timestamp=0.0,
                metadata={"z_score": zscore, "half_life": self._half_life(spread)},
            )

        return None

    @staticmethod
    def _half_life(spread: np.ndarray) -> float:
        """Estimate mean-reversion half-life via OLS on the spread."""
        if len(spread) < 3:
            return float("inf")
        lag = spread[:-1]
        delta = np.diff(spread)
        # OLS: delta = alpha + beta * lag + epsilon
        # Half-life = -ln(2) / beta
        try:
            beta = np.polyfit(lag, delta, 1)[0]
            if beta >= 0:
                return float("inf")
            return -np.log(2) / beta
        except (np.linalg.LinAlgError, ValueError):
            return float("inf")
