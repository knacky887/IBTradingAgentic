"""
TradingApp — Base Strategy

Abstract base class for all quantitative strategies.
Every strategy must implement generate_signal().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.core.models import Bar, Signal


class BaseStrategy(ABC):
    """
    Abstract base for all strategies.

    Subclasses implement:
      • generate_signal(symbol, bars) → Signal | None
        - Receives the full available bar history for a symbol.
        - Returns a Signal with direction and conviction, or None if no signal.
    """

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier (snake_case)."""

    @property
    def family(self) -> str:
        """Strategy family for grouping."""
        return "unknown"

    @abstractmethod
    def generate_signal(self, symbol: str, bars: list[Bar]) -> Signal | None:
        """
        Evaluate the latest market data and produce a signal.

        Args:
            symbol: The ticker being evaluated.
            bars:   Historical bars up to (and including) the current bar.
                    bars[-1] is the most recent bar.

        Returns:
            A Signal object if a trade opportunity is detected, otherwise None.
        """

    def _closes(self, bars: list[Bar]) -> list[float]:
        """Helper: extract close prices."""
        return [b.close for b in bars]

    def _volumes(self, bars: list[Bar]) -> list[float]:
        """Helper: extract volumes."""
        return [b.volume for b in bars]
