"""
TradingApp — Portfolio Management Agent

Aggregates signals from strategies into a target portfolio,
computes the delta vs current holdings, and proposes orders.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.core.base_agent import BaseAgent
from src.core.message_bus import MessageBus
from src.core.models import Signal, Order, OrderSide, OrderType, Fill, Direction

logger = logging.getLogger("tradingapp.agents.portfolio_management")


class PortfolioManagementAgent(BaseAgent):
    """
    Responsible for:
      1. Collecting signals from all strategies
      2. Combining signals into target portfolio weights
      3. Comparing target vs. current to generate proposed orders
      4. Reconciling fills to update current holdings
    """

    def __init__(
        self,
        bus: MessageBus,
        nav: float = 1_000_000.0,
        max_positions: int = 50,
        signal_combination: str = "conviction",  # equal | conviction | inv_vol
    ):
        super().__init__(name="portfolio_management", bus=bus)
        self._nav = nav
        self._max_positions = max_positions
        self._signal_combination = signal_combination

        # State
        self._pending_signals: dict[str, list[Signal]] = {}  # symbol → signals
        self._current_positions: dict[str, int] = {}           # symbol → shares held
        self._last_prices: dict[str, float] = {}               # symbol → last known price

    async def on_start(self) -> None:
        await self.subscribe("signals.*.*", self._on_signal)
        await self.subscribe("fills.*", self._on_fill)
        await self.subscribe("market.bars.*", self._on_bar)
        self.logger.info("Portfolio Management agent started — NAV=%.0f", self._nav)

    # ── Signal Aggregation ─────────────────────────────────────────────

    async def _on_signal(self, channel: str, payload: dict[str, Any]) -> None:
        """Collect incoming signals for aggregation."""
        signal = Signal(**payload)
        if signal.symbol not in self._pending_signals:
            self._pending_signals[signal.symbol] = []
        self._pending_signals[signal.symbol].append(signal)

    async def _on_bar(self, channel: str, payload: dict[str, Any]) -> None:
        """Track last prices for position sizing."""
        symbol = payload.get("symbol", "")
        close = payload.get("close", 0)
        if symbol and close:
            self._last_prices[symbol] = close

    async def _on_fill(self, channel: str, payload: dict[str, Any]) -> None:
        """Reconcile fills to update current holdings."""
        fill = Fill(**payload)
        delta = fill.filled_quantity if fill.side == OrderSide.BUY else -fill.filled_quantity
        self._current_positions[fill.symbol] = (
            self._current_positions.get(fill.symbol, 0) + delta
        )
        self.logger.info(
            "Position updated: %s → %d shares",
            fill.symbol, self._current_positions[fill.symbol],
        )

    # ── Portfolio Construction ─────────────────────────────────────────

    async def rebalance(self) -> None:
        """
        Combine pending signals into target portfolio and propose orders.
        Call this on a schedule (e.g., EOD) or after signal batch.
        """
        if not self._pending_signals:
            return

        target_weights = self._compute_target_weights()
        orders = self._generate_orders(target_weights)

        for order in orders:
            await self.publish("orders.proposed", order.model_dump())
            self.logger.info(
                "Proposed: %s %s %d @ %.2f",
                order.side.value, order.symbol, order.quantity,
                order.limit_price or 0,
            )

        # Clear processed signals
        self._pending_signals.clear()

    def _compute_target_weights(self) -> dict[str, float]:
        """
        Combine all pending signals into target weights per symbol.
        Returns {symbol: weight} where weight is signed (-1 to +1).
        """
        weights: dict[str, float] = {}

        def _direction_sign(d: Direction) -> int:
            if d == Direction.LONG:
                return 1
            elif d == Direction.SHORT:
                return -1
            return 0  # FLAT

        for symbol, signals in self._pending_signals.items():
            if not signals:
                continue

            if self._signal_combination == "equal":
                # Simple average direction
                raw = sum(
                    s.conviction * _direction_sign(s.direction)
                    for s in signals
                ) / len(signals)
            elif self._signal_combination == "conviction":
                # Conviction-weighted
                total_conv = sum(s.conviction for s in signals)
                raw = sum(
                    s.conviction * _direction_sign(s.direction)
                    for s in signals
                ) / max(total_conv, 1e-9)
            else:
                # Default: equal
                raw = sum(
                    s.conviction * _direction_sign(s.direction)
                    for s in signals
                ) / len(signals)

            weights[symbol] = max(-1.0, min(1.0, raw))

        # Limit to top N by absolute weight
        if len(weights) > self._max_positions:
            sorted_symbols = sorted(weights, key=lambda s: abs(weights[s]), reverse=True)
            weights = {s: weights[s] for s in sorted_symbols[:self._max_positions]}

        return weights

    def _generate_orders(self, target_weights: dict[str, float]) -> list[Order]:
        """Convert target weights into proposed orders."""
        orders = []

        for symbol, weight in target_weights.items():
            price = self._last_prices.get(symbol, 0)
            if price <= 0:
                continue

            target_notional = self._nav * weight * (1.0 / max(len(target_weights), 1))
            target_shares = int(target_notional / price)
            current_shares = self._current_positions.get(symbol, 0)
            delta = target_shares - current_shares

            if delta == 0:
                continue

            order = Order(
                symbol=symbol,
                side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                quantity=abs(delta),
                order_type=OrderType.LIMIT,
                limit_price=price,
                strategy="portfolio_rebalance",
                timestamp=time.time(),
            )
            orders.append(order)

        return orders
