"""
TradingApp — Tests for Data Ingestion and Execution (with IB mocking)

These tests mock ib_insync to exercise the IB-dependent code paths
without requiring a live broker connection.
"""

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.core.message_bus import InProcessMessageBus
from src.core.models import Bar, Order, Fill, OrderSide, OrderType, OrderStatus
from src.agents.data_ingestion import DataIngestionAgent
from src.agents.execution import ExecutionAgent


# ── Data Ingestion Tests ──────────────────────────────────────────────

class TestDataIngestionAgent:
    @pytest.mark.asyncio
    async def test_on_start_subscribes(self):
        bus = InProcessMessageBus()
        await bus.connect()

        agent = DataIngestionAgent(bus=bus, symbols=["AAPL", "MSFT"])
        # on_start in backtest mode (no IB) should just set up
        await agent.on_start()
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_on_stop_disconnects_ib(self):
        bus = InProcessMessageBus()
        await bus.connect()

        agent = DataIngestionAgent(bus=bus, symbols=["AAPL"])
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.disconnect = MagicMock()
        agent._ib = mock_ib

        await agent.on_stop()
        mock_ib.disconnect.assert_called_once()
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_on_stop_without_ib(self):
        bus = InProcessMessageBus()
        await bus.connect()
        agent = DataIngestionAgent(bus=bus, symbols=["AAPL"])
        await agent.on_stop()  # No IB connected, should not raise
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_bar_update_handler_publishes(self):
        bus = InProcessMessageBus()
        await bus.connect()

        bars_received = []
        async def capture_bar(ch, payload):
            bars_received.append(payload)

        await bus.subscribe("market.bars.*", capture_bar)

        agent = DataIngestionAgent(bus=bus, symbols=["AAPL"])

        # Create a mock RealTimeBar
        mock_bar = MagicMock()
        mock_bar.time = MagicMock()
        mock_bar.time.timestamp.return_value = 1700000000.0
        mock_bar.open_ = 150.0
        mock_bar.high = 152.0
        mock_bar.low = 148.0
        mock_bar.close = 151.0
        mock_bar.volume = 5000.0
        mock_bar.wap = 150.5

        mock_bars = MagicMock()
        mock_bars.__getitem__ = MagicMock(return_value=mock_bar)
        mock_bars.__len__ = MagicMock(return_value=1)

        # Invoke handler directly via asyncio
        await agent._on_bar_update("AAPL", mock_bars)

        assert len(bars_received) == 1
        assert bars_received[0]["symbol"] == "AAPL"
        assert bars_received[0]["close"] == 151.0
        await bus.disconnect()


# ── Execution Agent IB Path Tests ─────────────────────────────────────

class TestExecutionAgentIB:
    @pytest.mark.asyncio
    async def test_submit_limit_order_to_ib(self):
        bus = InProcessMessageBus()
        await bus.connect()

        agent = ExecutionAgent(bus=bus)
        await agent.on_start()

        # Mock IB
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.placeOrder = MagicMock()
        agent._ib = mock_ib

        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.LIMIT, limit_price=150.0,
            status=OrderStatus.APPROVED, timestamp=time.time(),
        )

        # Patch the monitor task to do nothing
        with patch.object(agent, '_monitor_ib_fill', new_callable=AsyncMock):
            await agent._submit_to_ib(order)

        mock_ib.placeOrder.assert_called_once()
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_submit_market_order_to_ib(self):
        bus = InProcessMessageBus()
        await bus.connect()

        agent = ExecutionAgent(bus=bus)
        await agent.on_start()

        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.placeOrder = MagicMock()
        agent._ib = mock_ib

        order = Order(
            symbol="MSFT", side=OrderSide.SELL, quantity=50,
            order_type=OrderType.MARKET,
            status=OrderStatus.APPROVED, timestamp=time.time(),
        )

        with patch.object(agent, '_monitor_ib_fill', new_callable=AsyncMock):
            await agent._submit_to_ib(order)

        mock_ib.placeOrder.assert_called_once()
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_monitor_ib_fill_completed(self):
        bus = InProcessMessageBus()
        await bus.connect()

        fills = []
        async def capture(ch, payload):
            fills.append(payload)
        await bus.subscribe("fills.*", capture)

        agent = ExecutionAgent(bus=bus)
        await agent.on_start()

        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.LIMIT, limit_price=150.0,
        )
        agent._open_orders[order.id] = order

        # Mock trade
        mock_trade = MagicMock()
        call_count = [0]
        def is_done():
            call_count[0] += 1
            return call_count[0] >= 2
        mock_trade.isDone = is_done

        mock_status = MagicMock()
        mock_status.status = "Filled"
        mock_status.filled = 100
        mock_status.avgFillPrice = 150.5
        mock_trade.orderStatus = mock_status
        mock_trade.commissionReportEvent = []

        await agent._monitor_ib_fill(order, mock_trade)

        assert len(fills) == 1
        assert fills[0]["filled_quantity"] == 100
        assert fills[0]["avg_price"] == 150.5
        assert order.id not in agent._open_orders
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_monitor_ib_fill_not_filled(self):
        bus = InProcessMessageBus()
        await bus.connect()

        fills = []
        async def capture(ch, payload):
            fills.append(payload)
        await bus.subscribe("fills.*", capture)

        agent = ExecutionAgent(bus=bus)
        await agent.on_start()

        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=100,
        )
        agent._open_orders[order.id] = order

        mock_trade = MagicMock()
        mock_trade.isDone = MagicMock(return_value=True)
        mock_status = MagicMock()
        mock_status.status = "Cancelled"
        mock_trade.orderStatus = mock_status

        await agent._monitor_ib_fill(order, mock_trade)

        assert len(fills) == 0  # No fill for cancelled
        assert order.id not in agent._open_orders
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_halt_with_ib_calls_global_cancel(self):
        bus = InProcessMessageBus()
        await bus.connect()

        agent = ExecutionAgent(bus=bus)
        await agent.on_start()

        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.reqGlobalCancel = MagicMock()
        agent._ib = mock_ib

        await bus.publish("system.halt_trading", {"reason": "TEST"})

        mock_ib.reqGlobalCancel.assert_called_once()
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_on_stop_with_ib(self):
        bus = InProcessMessageBus()
        await bus.connect()

        agent = ExecutionAgent(bus=bus)
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.disconnect = MagicMock()
        agent._ib = mock_ib

        await agent.on_stop()
        mock_ib.disconnect.assert_called_once()
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_approved_order_routes_to_ib_when_connected(self):
        bus = InProcessMessageBus()
        await bus.connect()

        fills = []
        async def capture(ch, payload):
            fills.append(payload)
        await bus.subscribe("fills.*", capture)

        agent = ExecutionAgent(bus=bus)
        await agent.on_start()

        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.placeOrder = MagicMock()
        agent._ib = mock_ib

        order = Order(
            symbol="GOOG", side=OrderSide.BUY, quantity=10,
            order_type=OrderType.LIMIT, limit_price=2800.0,
            status=OrderStatus.APPROVED, timestamp=time.time(),
        )

        with patch.object(agent, '_monitor_ib_fill', new_callable=AsyncMock):
            await bus.publish("orders.approved", order.model_dump())

        mock_ib.placeOrder.assert_called_once()
        assert len(fills) == 0  # Goes to IB, not simulated
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_monitor_fill_with_slippage_and_commission(self):
        bus = InProcessMessageBus()
        await bus.connect()

        fills = []
        async def capture(ch, payload):
            fills.append(payload)
        await bus.subscribe("fills.*", capture)

        agent = ExecutionAgent(bus=bus)
        await agent.on_start()

        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.LIMIT, limit_price=150.0,
        )

        mock_trade = MagicMock()
        mock_trade.isDone = MagicMock(return_value=True)
        mock_status = MagicMock()
        mock_status.status = "Filled"
        mock_status.filled = 100
        mock_status.avgFillPrice = 150.25
        mock_trade.orderStatus = mock_status

        # Mock commission reports
        mock_report = MagicMock()
        mock_report.commission = 2.50
        mock_trade.commissionReportEvent = [mock_report]

        await agent._monitor_ib_fill(order, mock_trade)

        assert len(fills) == 1
        assert fills[0]["commission"] == 2.50
        assert fills[0]["slippage_bps"] > 0  # Should have slippage
        await bus.disconnect()
