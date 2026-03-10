"""
TradingApp — Unit Tests for Risk Management Agent
"""

import asyncio
import pytest
from src.core.message_bus import InProcessMessageBus
from src.core.models import Order, OrderSide, OrderType, OrderStatus
from src.agents.risk_management import RiskManagementAgent


class TestRiskManagement:
    @pytest.fixture
    async def setup(self):
        bus = InProcessMessageBus()
        await bus.connect()
        agent = RiskManagementAgent(
            bus=bus,
            nav=100_000,
            max_position_pct=0.05,
            max_daily_drawdown_pct=0.02,
        )
        await agent.on_start()

        results = {"approved": [], "rejected": []}

        async def on_approved(ch, payload):
            results["approved"].append(payload)

        async def on_rejected(ch, payload):
            results["rejected"].append(payload)

        await bus.subscribe("orders.approved", on_approved)
        await bus.subscribe("orders.rejected", on_rejected)

        return bus, agent, results

    @pytest.mark.asyncio
    async def test_order_within_limits_approved(self, setup):
        bus, agent, results = setup

        # Order for 4% of NAV (under 5% limit)
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=26,
            order_type=OrderType.LIMIT, limit_price=150.0,
        )
        await bus.publish("orders.proposed", order.model_dump())

        assert len(results["approved"]) == 1
        assert len(results["rejected"]) == 0
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_order_exceeding_position_limit_rejected(self, setup):
        bus, agent, results = setup

        # Order for 10% of NAV (over 5% limit)
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.LIMIT, limit_price=150.0,
        )
        await bus.publish("orders.proposed", order.model_dump())

        assert len(results["rejected"]) == 1
        assert "POSITION_SIZE_EXCEEDED" in results["rejected"][0]["metadata"]["rejection_reason"]
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_sell_order_reduces_position(self, setup):
        """Bug 3 regression: SELL orders should reduce position, not increase it."""
        bus, agent, results = setup

        # Simulate existing long position of 4% NAV
        agent._positions["AAPL"] = 4000.0  # $4k notional

        # SELL order should reduce it, not be rejected
        order = Order(
            symbol="AAPL", side=OrderSide.SELL, quantity=20,
            order_type=OrderType.LIMIT, limit_price=150.0,
        )
        await bus.publish("orders.proposed", order.model_dump())

        assert len(results["approved"]) == 1
        assert len(results["rejected"]) == 0
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_trading_halt_rejects_all(self, setup):
        bus, agent, results = setup

        # Force halt
        agent._trading_halted = True

        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=1,
            order_type=OrderType.LIMIT, limit_price=150.0,
        )
        await bus.publish("orders.proposed", order.model_dump())

        assert len(results["rejected"]) == 1
        assert results["rejected"][0]["metadata"]["rejection_reason"] == "TRADING_HALTED"
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_profitable_day_does_not_trigger_drawdown(self, setup):
        """Bug 5 regression: positive PnL should NOT trigger drawdown limit."""
        bus, agent, results = setup

        # Set positive daily PnL (profitable day)
        agent._daily_pnl = 5000.0  # +$5k profit

        order = Order(
            symbol="MSFT", side=OrderSide.BUY, quantity=10,
            order_type=OrderType.LIMIT, limit_price=300.0,
        )
        await bus.publish("orders.proposed", order.model_dump())

        assert len(results["approved"]) == 1
        assert len(results["rejected"]) == 0
        await bus.disconnect()
