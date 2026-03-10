"""
TradingApp — Dual Moving Average Crossover Strategy

Classic trend-following strategy using EMA crossover with ADX regime filter.
  • Long  when fast EMA crosses above slow EMA and ADX > threshold
  • Short when fast EMA crosses below slow EMA and ADX > threshold
  • No signal when ADX < threshold (choppy market)
"""

from __future__ import annotations

import numpy as np

from src.core.models import Bar, Signal, Direction
from src.strategies.base_strategy import BaseStrategy


class DualMaCrossover(BaseStrategy):

    @property
    def name(self) -> str:
        return "dual_ma_crossover"

    @property
    def family(self) -> str:
        return "trend_following"

    def generate_signal(self, symbol: str, bars: list[Bar]) -> Signal | None:
        fast = self.params.get("fast_period", 20)
        slow = self.params.get("slow_period", 50)
        adx_threshold = self.params.get("adx_threshold", 25)
        adx_period = self.params.get("adx_period", 14)

        # Need enough bars
        min_bars = slow + adx_period + 1
        if len(bars) < min_bars:
            return None

        closes = np.array(self._closes(bars))
        highs = np.array([b.high for b in bars])
        lows = np.array([b.low for b in bars])

        # Compute EMAs
        fast_ema = self._ema(closes, fast)
        slow_ema = self._ema(closes, slow)

        # Crossover detection (current and previous)
        curr_fast, prev_fast = fast_ema[-1], fast_ema[-2]
        curr_slow, prev_slow = slow_ema[-1], slow_ema[-2]

        bullish_cross = prev_fast <= prev_slow and curr_fast > curr_slow
        bearish_cross = prev_fast >= prev_slow and curr_fast < curr_slow

        if not bullish_cross and not bearish_cross:
            return None

        # ADX filter
        adx = self._compute_adx(highs, lows, closes, adx_period)
        if adx < adx_threshold:
            return None

        direction = Direction.LONG if bullish_cross else Direction.SHORT
        conviction = min(1.0, adx / 50.0)  # Scale ADX to 0-1

        return Signal(
            strategy=self.name,
            symbol=symbol,
            direction=direction,
            conviction=conviction,
            timestamp=0.0,  # Set by agent
            metadata={"fast_ema": curr_fast, "slow_ema": curr_slow, "adx": adx},
        )

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Compute Exponential Moving Average."""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    @staticmethod
    def _compute_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float:
        """Compute Average Directional Index (current value)."""
        n = len(closes)
        if n < period + 1:
            return 0.0

        # True Range
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )

        # Directional Movement
        up_move = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed averages (Wilder's smoothing)
        atr = np.zeros(len(tr))
        plus_di_arr = np.zeros(len(tr))
        minus_di_arr = np.zeros(len(tr))

        atr[period - 1] = np.mean(tr[:period])
        plus_di_arr[period - 1] = np.mean(plus_dm[:period])
        minus_di_arr[period - 1] = np.mean(minus_dm[:period])

        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
            plus_di_arr[i] = (plus_di_arr[i - 1] * (period - 1) + plus_dm[i]) / period
            minus_di_arr[i] = (minus_di_arr[i - 1] * (period - 1) + minus_dm[i]) / period

        # DI values
        plus_di = 100 * plus_di_arr[-1] / max(atr[-1], 1e-10)
        minus_di = 100 * minus_di_arr[-1] / max(atr[-1], 1e-10)

        # DX → ADX
        dx = 100 * abs(plus_di - minus_di) / max(plus_di + minus_di, 1e-10)
        return dx  # Simplified: single-point ADX ≈ DX
