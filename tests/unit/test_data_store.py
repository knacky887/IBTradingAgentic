"""
TradingApp — Unit Tests for Data Store Utilities
"""

import pytest
from pathlib import Path

from src.core.models import Bar
from src.utils.data_store import bars_to_dataframe, dataframe_to_bars, save_parquet, load_parquet


def _make_bars(n=5, symbol="TEST"):
    return [
        Bar(
            symbol=symbol,
            timestamp=1700000000.0 + i * 60,
            open=100.0 + i, high=101.0 + i, low=99.0 + i,
            close=100.5 + i, volume=1000.0 + i,
        )
        for i in range(n)
    ]


class TestBarsToDataframe:
    def test_basic_conversion(self):
        bars = _make_bars(3)
        df = bars_to_dataframe(bars)
        assert len(df) == 3
        assert "close" in df.columns
        assert "symbol" in df.columns
        assert df.index.name == "datetime"

    def test_sorted_by_time(self):
        bars = _make_bars(5)
        df = bars_to_dataframe(bars)
        assert list(df["timestamp"]) == sorted(df["timestamp"])


class TestDataframeToBox:
    def test_roundtrip(self):
        original = _make_bars(4)
        df = bars_to_dataframe(original)
        restored = dataframe_to_bars(df)
        assert len(restored) == 4
        assert restored[0].symbol == "TEST"
        assert restored[0].close == 100.5


class TestSaveLoadParquet:
    def test_save_and_load(self, tmp_path):
        bars = _make_bars(10)
        filepath = save_parquet(bars, tmp_path, symbol="TEST")
        assert filepath.exists()
        loaded = load_parquet(filepath)
        assert len(loaded) == 10
        assert loaded[0].symbol == "TEST"

    def test_save_without_symbol(self, tmp_path):
        bars = _make_bars(3)
        filepath = tmp_path / "data.parquet"
        result = save_parquet(bars, filepath)
        assert result.exists()

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_parquet(tmp_path / "nope.parquet")
