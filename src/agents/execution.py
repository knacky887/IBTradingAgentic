"""
TradingApp — Execution Agent

Translates risk-approved orders into IB API calls, tracks order lifecycle,
and publishes fill confirmations.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from src.core.base_agent import BaseAgent
from src.core.message_bus import MessageBus
from src.core.models import Order, Fill, OrderSide, OrderStatus

logger = logging.getLogger("tradingapp.agents.execution")


class ExecutionAgent(BaseAgent):
    """
    Responsible for:
      1. Receiving approved orders from the Risk agent
      2. Submitting orders to IB via the TWS API
      3. Tracking order status (submitted → filled / cancelled)
      4. Publishing fill confirmations with slippage data
    """

    def __init__(
        self,
        bus: MessageBus,
        ib_host: str = "127.0.0.1",
        ib_port: int = 7497,
        ib_client_id: int = 2,
        max_slippage_bps: int = 10,
    ):
        super().__init__(name="execution", bus=bus)
        self._ib_host = ib_host
        self._ib_port = ib_port
        self._ib_client_id = ib_client_id
        self._max_slippage_bps = max_slippage_bps
        self._ib = None
        self._open_orders: dict[str, Order] = {}  # order_id → Order

    async def on_start(self) -> None:
        await self.subscribe("orders.approved", self._on_approved_order)
        await self.subscribe("system.halt_trading", self._on_halt)
        self.logger.info("Execution agent started.")
        try:
            await self._connect_ib()
        except Exception:
            self.logger.warning("IB not connected — execution will use simulated fills")

    async def on_stop(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()

    async def _connect_ib(self) -> None:
        from ib_insync import IB
        self._ib = IB()
        await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self._ib.connect(self._ib_host, self._ib_port, clientId=self._ib_client_id),
        )
        self.logger.info("Execution agent connected to IB.")

    # ── Order Handling ─────────────────────────────────────────────────

    async def _on_approved_order(self, channel: str, payload: dict[str, Any]) -> None:
        """Submit approved order to IB (or simulate fill)."""
        order = Order(**payload)
        self._open_orders[order.id] = order

        if self._ib and self._ib.isConnected():
            await self._submit_to_ib(order)
        else:
            # Simulated fill for backtest / paper mode without IB
            await self._simulate_fill(order)

    async def _submit_to_ib(self, order: Order) -> None:
        """Submit order to Interactive Brokers."""
        from ib_insync import Stock, LimitOrder, MarketOrder

        contract = Stock(order.symbol, "SMART", "USD")

        if order.order_type.value == "LIMIT" and order.limit_price:
            ib_order = LimitOrder(
                action=order.side.value,
                totalQuantity=order.quantity,
                lmtPrice=order.limit_price,
            )
        else:
            ib_order = MarketOrder(
                action=order.side.value,
                totalQuantity=order.quantity,
            )

        trade = self._ib.placeOrder(contract, ib_order)
        self.logger.info(
            "Submitted to IB: %s %s %d (order_id=%s)",
            order.side.value, order.symbol, order.quantity, order.id,
        )

        # Monitor fill in background
        asyncio.create_task(self._monitor_ib_fill(order, trade))

    async def _monitor_ib_fill(self, order: Order, trade: Any) -> None:
        """Wait for IB fill and publish confirmation."""
        while not trade.isDone():
            await asyncio.sleep(0.5)

        if trade.orderStatus.status == "Filled":
            fill = Fill(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                filled_quantity=int(trade.orderStatus.filled),
                avg_price=float(trade.orderStatus.avgFillPrice),
                commission=sum(c.commission for c in trade.commissionReportEvent) if trade.commissionReportEvent else 0.0,
                timestamp=time.time(),
            )
            # Compute slippage
            if order.limit_price:
                fill.slippage_bps = abs(fill.avg_price - order.limit_price) / order.limit_price * 10_000
            await self.publish(f"fills.{order.symbol}", fill.model_dump())
            self.logger.info("FILLED: %s %s %d @ %.2f", order.side.value, order.symbol, fill.filled_quantity, fill.avg_price)
        else:
            self.logger.warning("Order not filled — status: %s", trade.orderStatus.status)

        self._open_orders.pop(order.id, None)

    async def _simulate_fill(self, order: Order) -> None:
        """Simulate an immediate fill (backtest / no-IB mode)."""
        price = order.limit_price or 0
        fill = Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            filled_quantity=order.quantity,
            avg_price=price,
            commission=order.quantity * price * 0.0001,  # ~1 bps simulated commission
            slippage_bps=0.5,  # 0.5 bps simulated slippage
            timestamp=time.time(),
        )
        await self.publish(f"fills.{order.symbol}", fill.model_dump())
        self.logger.info(
            "SIM FILL: %s %s %d @ %.2f",
            order.side.value, order.symbol, fill.filled_quantity, fill.avg_price,
        )
        self._open_orders.pop(order.id, None)

    # ── Kill Switch ────────────────────────────────────────────────────

    async def _on_halt(self, channel: str, payload: dict[str, Any]) -> None:
        """Cancel all open orders on halt signal."""
        self.logger.critical("HALT received — cancelling all open orders")
        if self._ib and self._ib.isConnected():
            self._ib.reqGlobalCancel()
        self._open_orders.clear()
