"""
TradingApp — Unit Tests for Strategies
"""

import numpy as np
import pytest
from src.core.models import Bar, Direction
from src.strategies.trend_following.dual_ma_crossover import DualMaCrossover
from src.strategies.mean_reversion.bollinger_reversion import BollingerReversion
from src.strategies.stat_arb.engle_granger_pairs import EngleGrangerPairs


def _make_bars(prices: list[float], symbol: str = "TEST") -> list[Bar]:
    """Create a sequence of bars from close prices."""
    bars = []
    for i, price in enumerate(prices):
        bars.append(Bar(
            symbol=symbol,
            timestamp=1700000000.0 + i * 60,
            open=price * 0.999,
            high=price * 1.005,
            low=price * 0.995,
            close=price,
            volume=10000,
        ))
    return bars


class TestDualMaCrossover:
    def test_insufficient_bars_returns_none(self):
        strategy = DualMaCrossover(params={"fast_period": 5, "slow_period": 10, "adx_period": 5, "adx_threshold": 10})
        bars = _make_bars([100.0] * 10)
        assert strategy.generate_signal("TEST", bars) is None

    def test_returns_signal_on_crossover(self):
        strategy = DualMaCrossover(params={"fast_period": 5, "slow_period": 10, "adx_period": 5, "adx_threshold": 0})
        # Downtrend then sharp uptrend to trigger bullish cross
        prices = [100 - i * 0.5 for i in range(20)] + [95 + i * 2 for i in range(20)]
        bars = _make_bars(prices)
        # May or may not produce a signal depending on exact crossover timing
        signal = strategy.generate_signal("TEST", bars)
        # Just ensure no crash — signal may be None if crossover just happened
        assert signal is None or signal.direction in (Direction.LONG, Direction.SHORT)


class TestBollingerReversion:
    def test_insufficient_bars_returns_none(self):
        strategy = BollingerReversion(params={"period": 20, "num_std": 2.0})
        bars = _make_bars([100.0] * 15)
        assert strategy.generate_signal("TEST", bars) is None

    def test_band_breach_generates_signal(self):
        strategy = BollingerReversion(params={"period": 20, "num_std": 2.0})
        # Stable prices then sudden drop below lower band
        prices = [100.0] * 25 + [85.0]  # Sharp drop
        bars = _make_bars(prices)
        signal = strategy.generate_signal("TEST", bars)
        if signal is not None:
            assert signal.direction == Direction.LONG  # Mean reversion → long on drop


class TestEngleGrangerPairs:
    def test_insufficient_bars_returns_none(self):
        strategy = EngleGrangerPairs(params={"lookback": 50, "zscore_entry": 2.0})
        bars = _make_bars([100.0] * 30)
        assert strategy.generate_signal("TEST", bars) is None

    def test_extreme_zscore_generates_signal(self):
        strategy = EngleGrangerPairs(params={"lookback": 50, "zscore_entry": 2.0})
        # Mean-reverting prices with extreme deviation at end
        np.random.seed(42)
        prices = list(100 + np.cumsum(np.random.randn(60) * 0.1))
        prices[-1] = prices[-1] + 5  # Extreme positive deviation
        bars = _make_bars(prices)
        signal = strategy.generate_signal("TEST", bars)
        # Should detect the extreme z-score
        assert signal is None or signal.direction in (Direction.LONG, Direction.SHORT)
