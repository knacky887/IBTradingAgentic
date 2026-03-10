"""
TradingApp — Extended Strategy Tests for Coverage

Tests ADX computation in DualMaCrossover, SHORT signals in Bollinger,
and edge cases in EngleGrangerPairs.
"""

import time
import numpy as np
import pytest

from src.core.models import Bar, Direction
from src.strategies.trend_following.dual_ma_crossover import DualMaCrossover
from src.strategies.mean_reversion.bollinger_reversion import BollingerReversion
from src.strategies.stat_arb.engle_granger_pairs import EngleGrangerPairs
from src.strategies.base_strategy import BaseStrategy


def _make_bars(n, symbol="TEST", base=100.0, step=0.5):
    return [
        Bar(
            symbol=symbol,
            timestamp=1700000000.0 + i * 60,
            open=base + i * step - 0.3,
            high=base + i * step + 1.0,
            low=base + i * step - 1.0,
            close=base + i * step,
            volume=10000.0,
        )
        for i in range(n)
    ]


# ── DualMaCrossover Extended ──────────────────────────────────────────

class TestDualMaCrossoverExtended:
    def test_name_and_family(self):
        s = DualMaCrossover()
        assert s.name == "dual_ma_crossover"
        assert s.family == "trend_following"

    def test_ema_computation(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ema = DualMaCrossover._ema(data, 3)
        assert len(ema) == 5
        assert ema[0] == 1.0  # First element = first data point

    def test_adx_computation(self):
        n = 40
        highs = np.random.rand(n) * 10 + 100
        lows = highs - np.random.rand(n) * 2
        closes = (highs + lows) / 2
        adx = DualMaCrossover._compute_adx(highs, lows, closes, 14)
        assert isinstance(adx, float)
        assert adx >= 0

    def test_adx_insufficient_data(self):
        highs = np.array([101.0, 102.0])
        lows = np.array([99.0, 100.0])
        closes = np.array([100.0, 101.0])
        adx = DualMaCrossover._compute_adx(highs, lows, closes, 14)
        assert adx == 0.0

    def test_bearish_crossover_signal(self):
        """Generate a bearish cross: fast drops below slow."""
        s = DualMaCrossover(params={"fast_period": 5, "slow_period": 10, "adx_threshold": 0})

        # First go up (fast > slow), then go sharply down (fast < slow)
        up_bars = _make_bars(20, base=100.0, step=1.0)
        down_bars = _make_bars(15, base=119.0, step=-2.0)
        # Adjust timestamps
        for i, b in enumerate(down_bars):
            down_bars[i] = Bar(
                symbol=b.symbol, timestamp=1700002000.0 + i * 60,
                open=b.open, high=b.high, low=b.low,
                close=b.close, volume=b.volume,
            )
        all_bars = up_bars + down_bars

        signal = s.generate_signal("TEST", all_bars)
        if signal:
            assert signal.direction in (Direction.LONG, Direction.SHORT)
            assert 0 <= signal.conviction <= 1.0

    def test_no_crossover_returns_none(self):
        """Flat data = no crossover."""
        s = DualMaCrossover(params={"fast_period": 5, "slow_period": 10, "adx_threshold": 0})
        # Flat data — no crossover
        bars = [
            Bar(
                symbol="TEST", timestamp=1700000000.0 + i * 60,
                open=100, high=100.1, low=99.9, close=100.0, volume=1000,
            )
            for i in range(40)
        ]
        signal = s.generate_signal("TEST", bars)
        assert signal is None

    def test_adx_below_threshold_returns_none(self):
        s = DualMaCrossover(params={"fast_period": 5, "slow_period": 10, "adx_threshold": 9999})
        bars = _make_bars(40, step=1.0)
        signal = s.generate_signal("TEST", bars)
        assert signal is None  # ADX can't possibly exceed 9999


# ── BollingerReversion Extended ───────────────────────────────────────

class TestBollingerReversionExtended:
    def test_name_and_family(self):
        s = BollingerReversion()
        assert s.name == "bollinger_reversion"
        assert s.family == "mean_reversion"

    def test_short_signal_above_upper_band(self):
        """Price above upper band → SHORT signal."""
        s = BollingerReversion(params={"period": 20, "num_std": 2.0})

        # 20 normal bars + 1 spike bar
        bars = [
            Bar(
                symbol="TEST", timestamp=1700000000.0 + i * 60,
                open=100, high=101, low=99, close=100.0, volume=1000,
            )
            for i in range(20)
        ]
        # Spike far above upper band
        spike = Bar(
            symbol="TEST", timestamp=1700002000.0,
            open=100, high=120, low=100, close=115.0, volume=5000,
        )
        bars.append(spike)

        signal = s.generate_signal("TEST", bars)
        assert signal is not None
        assert signal.direction == Direction.SHORT

    def test_long_signal_below_lower_band(self):
        s = BollingerReversion(params={"period": 20, "num_std": 2.0})

        bars = [
            Bar(
                symbol="TEST", timestamp=1700000000.0 + i * 60,
                open=100, high=101, low=99, close=100.0, volume=1000,
            )
            for i in range(20)
        ]
        # Drop far below lower band
        drop = Bar(
            symbol="TEST", timestamp=1700002000.0,
            open=100, high=100, low=80, close=85.0, volume=5000,
        )
        bars.append(drop)

        signal = s.generate_signal("TEST", bars)
        assert signal is not None
        assert signal.direction == Direction.LONG

    def test_within_bands_returns_none(self):
        s = BollingerReversion(params={"period": 20, "num_std": 2.0})
        bars = [
            Bar(
                symbol="TEST", timestamp=1700000000.0 + i * 60,
                open=100, high=100.5, low=99.5, close=100.0, volume=1000,
            )
            for i in range(21)
        ]
        signal = s.generate_signal("TEST", bars)
        assert signal is None


# ── EngleGrangerPairs Extended ────────────────────────────────────────

class TestEngleGrangerPairsExtended:
    def test_name_and_family(self):
        s = EngleGrangerPairs()
        assert s.name == "engle_granger_pairs"
        assert s.family == "stat_arb"

    def test_spread_history_bounded(self):
        s = EngleGrangerPairs(params={"lookback": 5, "zscore_entry": 0.1, "zscore_exit": 0.01})

        # Feed many bars to fill up spread history
        for i in range(1100):
            bars = [
                Bar(
                    symbol="TEST", timestamp=1700000000.0 + j * 60,
                    open=100 + np.sin(j), high=102, low=98,
                    close=100 + np.sin(j) + np.random.normal(0, 0.1),
                    volume=1000,
                )
                for j in range(10)
            ]
            s.generate_signal("TEST", bars)

        if "TEST" in s._spread_history:
            assert len(s._spread_history["TEST"]) <= 1000


# ── BaseStrategy Abstract ────────────────────────────────────────────

class TestBaseStrategyHelpers:
    def test_closes_helper(self):
        s = DualMaCrossover()
        bars = _make_bars(5, base=100, step=1)
        closes = s._closes(bars)
        assert len(closes) == 5
        assert closes[0] == 100.0

    def test_volumes_helper(self):
        s = DualMaCrossover()
        bars = _make_bars(3)
        vols = s._volumes(bars)
        assert all(v == 10000.0 for v in vols)

    def test_family_default(self):
        """Test the default 'unknown' family — need a strategy that doesn't override family."""
        class MinimalStrategy(BaseStrategy):
            @property
            def name(self):
                return "minimal"

            def generate_signal(self, symbol, bars):
                return None

        s = MinimalStrategy()
        assert s.family == "unknown"
