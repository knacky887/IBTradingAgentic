"""
TradingApp — Risk Management Agent

Pre-trade gatekeeper and real-time portfolio risk monitor.
Every proposed order must pass all risk checks before approval.
"""

from __future__ import annotations

import logging
import time
from typing import Any



from src.core.base_agent import BaseAgent
from src.core.message_bus import MessageBus
from src.core.models import Order, OrderStatus, OrderSide, Fill

logger = logging.getLogger("tradingapp.agents.risk_management")


class RiskManagementAgent(BaseAgent):
    """
    Responsible for:
      1. Pre-trade risk checks on every proposed order
      2. Real-time portfolio risk monitoring (VaR, drawdown)
      3. Kill-switch: halt all trading if drawdown exceeds threshold
    """

    def __init__(
        self,
        bus: MessageBus,
        nav: float = 1_000_000.0,
        max_position_pct: float = 0.05,
        max_sector_pct: float = 0.25,
        max_daily_drawdown_pct: float = 0.02,
        max_correlated_positions: int = 5,
        adv_min_multiplier: float = 0.01,
    ):
        super().__init__(name="risk_management", bus=bus)
        self._nav = nav
        self._max_position_pct = max_position_pct
        self._max_sector_pct = max_sector_pct
        self._max_daily_drawdown_pct = max_daily_drawdown_pct
        self._max_correlated_positions = max_correlated_positions
        self._adv_min_multiplier = adv_min_multiplier

        # State
        self._positions: dict[str, float] = {}   # symbol → notional value
        self._daily_pnl: float = 0.0
        self._high_water_mark: float = nav
        self._trading_halted: bool = False

    async def on_start(self) -> None:
        await self.subscribe("orders.proposed", self._on_proposed_order)
        await self.subscribe("fills.*", self._on_fill)
        self.logger.info(
            "Risk agent started — NAV=%.0f, max_pos=%.1f%%, max_dd=%.1f%%",
            self._nav, self._max_position_pct * 100, self._max_daily_drawdown_pct * 100,
        )

    # ── Pre-Trade Risk Checks ──────────────────────────────────────────

    async def _on_proposed_order(self, channel: str, payload: dict[str, Any]) -> None:
        """Gate every proposed order through risk checks."""
        order = Order(**payload)

        if self._trading_halted:
            order.status = OrderStatus.REJECTED
            order.metadata["rejection_reason"] = "TRADING_HALTED"
            await self.publish("orders.rejected", order.model_dump())
            self.logger.warning("Order REJECTED (halted): %s %s %d", order.side.value, order.symbol, order.quantity)
            return

        rejection = self._check_risk(order)
        if rejection:
            order.status = OrderStatus.REJECTED
            order.metadata["rejection_reason"] = rejection
            await self.publish("orders.rejected", order.model_dump())
            self.logger.warning("Order REJECTED (%s): %s %s %d", rejection, order.side.value, order.symbol, order.quantity)
        else:
            order.status = OrderStatus.APPROVED
            await self.publish("orders.approved", order.model_dump())
            self.logger.info("Order APPROVED: %s %s %d", order.side.value, order.symbol, order.quantity)

    def _check_risk(self, order: Order) -> str | None:
        """
        Run all pre-trade risk checks. Returns rejection reason or None if OK.
        """
        price = order.limit_price or 0
        notional = order.quantity * price
        # SELL orders reduce exposure, not increase it
        signed_notional = -notional if order.side == OrderSide.SELL else notional

        # 1. Position size limit
        current_pos = self._positions.get(order.symbol, 0)
        new_pos = current_pos + signed_notional
        if abs(new_pos) > self._nav * self._max_position_pct:
            return f"POSITION_SIZE_EXCEEDED ({abs(new_pos):.0f} > {self._nav * self._max_position_pct:.0f})"

        # 2. Daily drawdown limit (only trigger on losses, not profits)
        if self._daily_pnl < -(self._nav * self._max_daily_drawdown_pct):
            return f"DAILY_DRAWDOWN_EXCEEDED ({self._daily_pnl:.0f})"

        # 3. Total exposure limit (sum of all positions vs NAV)
        total_exposure = sum(abs(v) for v in self._positions.values()) + abs(signed_notional)
        if total_exposure > self._nav * 1.5:  # 150% gross exposure cap
            return f"GROSS_EXPOSURE_EXCEEDED ({total_exposure:.0f})"

        return None

    # ── Fill Processing ────────────────────────────────────────────────

    async def _on_fill(self, channel: str, payload: dict[str, Any]) -> None:
        """Update position and PnL tracking on fills."""
        fill = Fill(**payload)
        sign = 1.0 if fill.side.value == "BUY" else -1.0
        notional = fill.filled_quantity * fill.avg_price * sign
        self._positions[fill.symbol] = self._positions.get(fill.symbol, 0) + notional

        # Track daily PnL (simplified — real PnL needs mark-to-market)
        self._daily_pnl -= fill.commission

        # Check for drawdown halt
        current_nav = self._nav + self._daily_pnl
        if current_nav > self._high_water_mark:
            self._high_water_mark = current_nav
        drawdown = (self._high_water_mark - current_nav) / self._high_water_mark

        if drawdown > self._max_daily_drawdown_pct:
            self._trading_halted = True
            await self.publish("system.halt_trading", {
                "reason": f"DRAWDOWN_{drawdown:.2%}",
                "timestamp": time.time(),
            })
            self.logger.critical("TRADING HALTED — drawdown %.2f%% exceeds limit", drawdown * 100)

    # ── Risk Reset (EOD) ───────────────────────────────────────────────

    def reset_daily(self) -> None:
        """Reset daily PnL counters — called at end of trading day."""
        self._daily_pnl = 0.0
        self._trading_halted = False
        self.logger.info("Daily risk counters reset.")
