"""
TradingApp — Alpha Generation Agent

Runs a registry of strategy modules, each producing a Signal when
conviction exceeds its threshold. Strategies are individually toggleable.
"""

from __future__ import annotations

import importlib
import logging
import time
from typing import Any

from src.core.base_agent import BaseAgent
from src.core.message_bus import MessageBus
from src.core.models import Bar, Signal

logger = logging.getLogger("tradingapp.agents.alpha_generation")


class AlphaGenerationAgent(BaseAgent):
    """
    Responsible for:
      1. Loading and managing a registry of strategy instances
      2. Receiving bars from market.bars.* and forwarding to strategies
      3. Publishing signals on signals.{strategy}.{symbol}
    """

    def __init__(
        self,
        bus: MessageBus,
        strategy_config: dict[str, Any],
    ):
        super().__init__(name="alpha_generation", bus=bus)
        self._strategy_config = strategy_config
        self._strategies: list[Any] = []    # list of BaseStrategy instances
        self._bar_history: dict[str, list[Bar]] = {}  # symbol → recent bars

    async def on_start(self) -> None:
        """Load enabled strategies and subscribe to market bars."""
        self._load_strategies()
        await self.subscribe("market.bars.*", self._on_bar)
        self.logger.info(
            "Alpha agent started with %d strategies enabled.", len(self._strategies)
        )

    def _load_strategies(self) -> None:
        """Dynamically load enabled strategy classes from config."""
        from src.strategies.base_strategy import BaseStrategy

        # Strategy family → module path mapping
        family_modules = {
            "trend_following": "src.strategies.trend_following",
            "mean_reversion": "src.strategies.mean_reversion",
            "stat_arb": "src.strategies.stat_arb",
            "cross_sectional": "src.strategies.cross_sectional",
            "volatility": "src.strategies.volatility",
        }

        for name, cfg in self._strategy_config.items():
            if not cfg.get("enabled", False):
                continue

            family = cfg.get("family", "")
            module_path = family_modules.get(family)
            if not module_path:
                self.logger.warning("Unknown strategy family '%s' for '%s'", family, name)
                continue

            try:
                module = importlib.import_module(f"{module_path}.{name}")
                # Convention: class name is CamelCase of the strategy name
                class_name = "".join(word.capitalize() for word in name.split("_"))
                strategy_class = getattr(module, class_name)
                instance = strategy_class(params=cfg.get("params", {}))

                if not isinstance(instance, BaseStrategy):
                    self.logger.warning("'%s' does not inherit BaseStrategy — skipping", name)
                    continue

                self._strategies.append(instance)
                self.logger.info("Loaded strategy: %s (%s)", name, family)

            except (ImportError, AttributeError) as e:
                self.logger.warning("Could not load strategy '%s': %s", name, e)

    async def _on_bar(self, channel: str, payload: dict[str, Any]) -> None:
        """Handle incoming bar data — run all strategies."""
        bar = Bar(**payload)
        symbol = bar.symbol

        # Append to history
        if symbol not in self._bar_history:
            self._bar_history[symbol] = []
        self._bar_history[symbol].append(bar)

        # Keep history bounded (max 1000 bars per symbol)
        if len(self._bar_history[symbol]) > 1000:
            self._bar_history[symbol] = self._bar_history[symbol][-1000:]

        # Run each strategy
        bars = self._bar_history[symbol]
        for strategy in self._strategies:
            try:
                signal = strategy.generate_signal(symbol, bars)
                if signal is not None:
                    signal.timestamp = time.time()
                    await self.publish(
                        f"signals.{signal.strategy}.{signal.symbol}",
                        signal.model_dump(),
                    )
                    self.logger.info(
                        "Signal: %s %s %s conviction=%.2f",
                        signal.strategy, signal.direction.value,
                        signal.symbol, signal.conviction,
                    )
            except Exception:
                self.logger.exception(
                    "Strategy '%s' failed on %s", strategy.name, symbol
                )
