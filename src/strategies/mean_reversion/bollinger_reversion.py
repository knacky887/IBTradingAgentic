"""
TradingApp — Bollinger Band Reversion Strategy

Mean-reversion strategy using ATR-scaled Bollinger Bands.
  • Long  when price crosses below the lower band
  • Short when price crosses above the upper band
  • Exit  when price returns to the moving average
"""

from __future__ import annotations

import numpy as np

from src.core.models import Bar, Signal, Direction
from src.strategies.base_strategy import BaseStrategy


class BollingerReversion(BaseStrategy):

    @property
    def name(self) -> str:
        return "bollinger_reversion"

    @property
    def family(self) -> str:
        return "mean_reversion"

    def generate_signal(self, symbol: str, bars: list[Bar]) -> Signal | None:
        period = self.params.get("period", 20)
        num_std = self.params.get("num_std", 2.0)

        if len(bars) < period + 1:
            return None

        closes = np.array(self._closes(bars))

        # Bollinger Bands
        sma = np.mean(closes[-period:])
        std = np.std(closes[-period:], ddof=1)

        upper = sma + num_std * std
        lower = sma - num_std * std

        curr_close = closes[-1]
        prev_close = closes[-2]

        # Entry signals: price crossing bands
        if prev_close >= lower and curr_close < lower:
            # Price broke below lower band → long (mean reversion)
            distance = (sma - curr_close) / std if std > 0 else 0
            conviction = min(1.0, abs(distance) / 3.0)
            return Signal(
                strategy=self.name,
                symbol=symbol,
                direction=Direction.LONG,
                conviction=conviction,
                timestamp=0.0,
                metadata={
                    "sma": sma, "upper": upper, "lower": lower,
                    "z_score": -distance,
                },
            )

        if prev_close <= upper and curr_close > upper:
            # Price broke above upper band → short (mean reversion)
            distance = (curr_close - sma) / std if std > 0 else 0
            conviction = min(1.0, abs(distance) / 3.0)
            return Signal(
                strategy=self.name,
                symbol=symbol,
                direction=Direction.SHORT,
                conviction=conviction,
                timestamp=0.0,
                metadata={
                    "sma": sma, "upper": upper, "lower": lower,
                    "z_score": distance,
                },
            )

        return None
