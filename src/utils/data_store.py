"""
TradingApp — Data Store Utilities

Parquet read/write helpers for long-term historical data storage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.core.models import Bar

logger = logging.getLogger("tradingapp.utils.data_store")


def bars_to_dataframe(bars: list[Bar]) -> pd.DataFrame:
    """Convert a list of Bar objects to a Pandas DataFrame."""
    records = [b.model_dump() for b in bars]
    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("datetime").sort_index()
    return df


def dataframe_to_bars(df: pd.DataFrame) -> list[Bar]:
    """Convert a DataFrame back to a list of Bar objects."""
    bars = []
    for _, row in df.iterrows():
        bars.append(Bar(
            symbol=row["symbol"],
            timestamp=row["timestamp"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            vwap=row.get("vwap"),
        ))
    return bars


def save_parquet(bars: list[Bar], path: str | Path, symbol: str | None = None) -> Path:
    """Save bars to a Parquet file."""
    path = Path(path)
    if symbol:
        path = path / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)

    df = bars_to_dataframe(bars)
    df.to_parquet(path, engine="pyarrow", index=True)
    logger.info("Saved %d bars to %s", len(bars), path)
    return path


def load_parquet(path: str | Path) -> list[Bar]:
    """Load bars from a Parquet file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No parquet file at {path}")

    df = pd.read_parquet(path, engine="pyarrow")

    # Ensure timestamp column exists
    if "timestamp" not in df.columns and df.index.name == "datetime":
        df["timestamp"] = df.index.astype("int64") / 1e9

    return dataframe_to_bars(df)
