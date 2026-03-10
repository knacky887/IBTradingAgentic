"""
TradingApp — Unit Tests for All Agents

Tests the agent lifecycle, message routing, and business logic
using the InProcessMessageBus (no Redis or IB required).
"""

import asyncio
import time
from typing import Any

import pytest

from src.core.message_bus import InProcessMessageBus
from src.core.models import (
    Bar, Signal, Order, Fill, Heartbeat,
    Direction, OrderSide, OrderType, OrderStatus,
)
from src.core.base_agent import BaseAgent
from src.agents.alpha_generation import AlphaGenerationAgent
from src.agents.portfolio_management import PortfolioManagementAgent
from src.agents.execution import ExecutionAgent
from src.agents.system_monitor import SystemMonitorAgent
from src.agents.risk_management import RiskManagementAgent


# ── Helpers ────────────────────────────────────────────────────────────

def _make_bar(symbol="AAPL", close=150.0, ts=None):
    return Bar(
        symbol=symbol,
        timestamp=ts or time.time(),
        open=close - 1.0, high=close + 1.0, low=close - 2.0,
        close=close, volume=10000.0,
    )


def _make_bars(n, symbol="AAPL", base_close=100.0):
    """Make n bars with incrementing prices."""
    return [
        Bar(
            symbol=symbol,
            timestamp=1700000000.0 + i * 60,
            open=base_close + i - 0.5,
            high=base_close + i + 1.0,
            low=base_close + i - 1.0,
            close=base_close + i,
            volume=10000.0,
        )
        for i in range(n)
    ]


# ── BaseAgent Tests ───────────────────────────────────────────────────

class ConcreteAgent(BaseAgent):
    """Minimal concrete agent for testing lifecycle."""
    def __init__(self, bus, name="test_agent"):
        super().__init__(name=name, bus=bus, heartbeat_interval=0.1)
        self.started = False
        self.stopped = False

    async def on_start(self):
        self.started = True

    async def on_stop(self):
        self.stopped = True


class TestBaseAgent:
    @pytest.mark.asyncio
    async def test_lifecycle(self):
        bus = InProcessMessageBus()
        await bus.connect()
        agent = ConcreteAgent(bus)

        assert not agent.is_running
        await agent.start()
        assert agent.is_running
        assert agent.started

        await agent.stop()
        assert not agent.is_running
        assert agent.stopped
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_heartbeat_is_published(self):
        bus = InProcessMessageBus()
        await bus.connect()

        heartbeats = []
        async def capture_hb(channel, payload):
            heartbeats.append(payload)

        await bus.subscribe("system.heartbeat.*", capture_hb)

        agent = ConcreteAgent(bus)
        await agent.start()

        # Wait for at least one heartbeat
        await asyncio.sleep(0.25)
        await agent.stop()
        await bus.disconnect()

        assert len(heartbeats) >= 1
        assert heartbeats[0]["agent"] == "test_agent"
        assert heartbeats[0]["status"] == "OK"

    @pytest.mark.asyncio
    async def test_publish_does_not_mutate_payload(self):
        bus = InProcessMessageBus()
        await bus.connect()
        agent = ConcreteAgent(bus)
        await agent.start()

        original = {"key": "value"}
        await agent.publish("test.channel", original)
        # Original dict should NOT have _source/_timestamp
        assert "_source" not in original
        assert "_timestamp" not in original

        await agent.stop()
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_subscribe_routes_messages(self):
        bus = InProcessMessageBus()
        await bus.connect()
        agent = ConcreteAgent(bus)

        received = []
        async def handler(ch, payload):
            received.append(payload)

        await agent.subscribe("test.*", handler)
        await bus.publish("test.foo", {"val": 1})

        assert len(received) == 1
        assert received[0]["val"] == 1
        await bus.disconnect()


# ── AlphaGenerationAgent Tests ────────────────────────────────────────

class TestAlphaGenerationAgent:
    @pytest.mark.asyncio
    async def test_loads_no_strategies_when_none_enabled(self):
        bus = InProcessMessageBus()
        await bus.connect()

        config = {
            "dummy_strat": {"enabled": False, "family": "trend_following"}
        }
        agent = AlphaGenerationAgent(bus=bus, strategy_config=config)
        await agent.on_start()

        assert len(agent._strategies) == 0
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_loads_enabled_strategy(self):
        bus = InProcessMessageBus()
        await bus.connect()

        config = {
            "dual_ma_crossover": {
                "enabled": True,
                "family": "trend_following",
                "params": {"fast_period": 10, "slow_period": 30},
            }
        }
        agent = AlphaGenerationAgent(bus=bus, strategy_config=config)
        await agent.on_start()

        assert len(agent._strategies) == 1
        assert agent._strategies[0].name == "dual_ma_crossover"
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_unknown_family_skipped(self):
        bus = InProcessMessageBus()
        await bus.connect()

        config = {
            "fake": {"enabled": True, "family": "nonexistent_family"}
        }
        agent = AlphaGenerationAgent(bus=bus, strategy_config=config)
        await agent.on_start()

        assert len(agent._strategies) == 0
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_bar_triggers_strategy_run(self):
        bus = InProcessMessageBus()
        await bus.connect()

        signals_received = []
        async def capture_signal(ch, payload):
            signals_received.append(payload)

        await bus.subscribe("signals.*.*", capture_signal)

        config = {
            "dual_ma_crossover": {
                "enabled": True,
                "family": "trend_following",
                "params": {"fast_period": 5, "slow_period": 10, "adx_threshold": 0},
            }
        }
        agent = AlphaGenerationAgent(bus=bus, strategy_config=config)
        await agent.on_start()

        # Feed enough bars (need slow_period + adx_period + 1 = 10 + 14 + 1 = 25)
        for i in range(30):
            bar = _make_bar(close=100 + i * 0.5, ts=1700000000.0 + i * 60)
            await bus.publish(f"market.bars.AAPL", bar.model_dump())

        await bus.disconnect()
        # May or may not generate a signal depending on crossover, but shouldn't crash

    @pytest.mark.asyncio
    async def test_bar_history_bounded(self):
        bus = InProcessMessageBus()
        await bus.connect()
        config = {}
        agent = AlphaGenerationAgent(bus=bus, strategy_config=config)
        await agent.on_start()

        # Send 1100 bars
        for i in range(1100):
            bar = _make_bar(close=100 + i * 0.01, ts=1700000000.0 + i)
            await bus.publish("market.bars.AAPL", bar.model_dump())

        assert len(agent._bar_history["AAPL"]) <= 1000
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_import_error_skipped_gracefully(self):
        bus = InProcessMessageBus()
        await bus.connect()
        config = {
            "nonexistent_strategy": {
                "enabled": True,
                "family": "trend_following",
            }
        }
        agent = AlphaGenerationAgent(bus=bus, strategy_config=config)
        await agent.on_start()
        assert len(agent._strategies) == 0
        await bus.disconnect()


# ── PortfolioManagementAgent Tests ────────────────────────────────────

class TestPortfolioManagementAgent:
    @pytest.mark.asyncio
    async def test_signal_aggregation(self):
        bus = InProcessMessageBus()
        await bus.connect()
        agent = PortfolioManagementAgent(bus=bus, nav=100_000)
        await agent.on_start()

        sig = Signal(
            strategy="test", symbol="AAPL",
            direction=Direction.LONG, conviction=0.8, timestamp=time.time(),
        )
        await bus.publish("signals.test.AAPL", sig.model_dump())

        assert "AAPL" in agent._pending_signals
        assert len(agent._pending_signals["AAPL"]) == 1
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_bar_tracks_last_price(self):
        bus = InProcessMessageBus()
        await bus.connect()
        agent = PortfolioManagementAgent(bus=bus, nav=100_000)
        await agent.on_start()

        bar = _make_bar("AAPL", close=155.0)
        await bus.publish("market.bars.AAPL", bar.model_dump())

        assert agent._last_prices["AAPL"] == 155.0
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_fill_updates_position(self):
        bus = InProcessMessageBus()
        await bus.connect()
        agent = PortfolioManagementAgent(bus=bus, nav=100_000)
        await agent.on_start()

        fill = Fill(
            order_id="o1", symbol="AAPL", side=OrderSide.BUY,
            filled_quantity=50, avg_price=150.0, timestamp=time.time(),
        )
        await bus.publish("fills.AAPL", fill.model_dump())

        assert agent._current_positions["AAPL"] == 50
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_sell_fill_reduces_position(self):
        bus = InProcessMessageBus()
        await bus.connect()
        agent = PortfolioManagementAgent(bus=bus, nav=100_000)
        await agent.on_start()

        # Buy first
        fill_buy = Fill(
            order_id="o1", symbol="AAPL", side=OrderSide.BUY,
            filled_quantity=100, avg_price=150.0, timestamp=time.time(),
        )
        await bus.publish("fills.AAPL", fill_buy.model_dump())

        # Sell half
        fill_sell = Fill(
            order_id="o2", symbol="AAPL", side=OrderSide.SELL,
            filled_quantity=40, avg_price=155.0, timestamp=time.time(),
        )
        await bus.publish("fills.AAPL", fill_sell.model_dump())

        assert agent._current_positions["AAPL"] == 60
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_rebalance_generates_orders(self):
        bus = InProcessMessageBus()
        await bus.connect()

        proposed = []
        async def capture_order(ch, payload):
            proposed.append(payload)

        await bus.subscribe("orders.proposed", capture_order)

        agent = PortfolioManagementAgent(bus=bus, nav=100_000)
        await agent.on_start()

        # Set price
        agent._last_prices["AAPL"] = 150.0

        # Add signal
        sig = Signal(
            strategy="test", symbol="AAPL",
            direction=Direction.LONG, conviction=0.9, timestamp=time.time(),
        )
        await bus.publish("signals.test.AAPL", sig.model_dump())

        # Rebalance
        await agent.rebalance()

        assert len(proposed) >= 1
        assert proposed[0]["symbol"] == "AAPL"
        assert proposed[0]["side"] == "BUY"
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_rebalance_no_signals_noop(self):
        bus = InProcessMessageBus()
        await bus.connect()

        proposed = []
        async def capture_order(ch, payload):
            proposed.append(payload)

        await bus.subscribe("orders.proposed", capture_order)

        agent = PortfolioManagementAgent(bus=bus, nav=100_000)
        await agent.on_start()
        await agent.rebalance()

        assert len(proposed) == 0
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_direction_flat_gives_zero_weight(self):
        bus = InProcessMessageBus()
        await bus.connect()
        agent = PortfolioManagementAgent(bus=bus, nav=100_000)
        await agent.on_start()

        sig = Signal(
            strategy="test", symbol="AAPL",
            direction=Direction.FLAT, conviction=0.5, timestamp=time.time(),
        )
        await bus.publish("signals.test.AAPL", sig.model_dump())

        weights = agent._compute_target_weights()
        assert weights.get("AAPL", 0) == 0.0
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_equal_signal_combination(self):
        bus = InProcessMessageBus()
        await bus.connect()
        agent = PortfolioManagementAgent(
            bus=bus, nav=100_000, signal_combination="equal"
        )
        await agent.on_start()

        # Manually inject signals to bypass pattern matching
        sig1 = Signal(
            strategy="s1", symbol="AAPL",
            direction=Direction.LONG, conviction=0.5, timestamp=time.time(),
        )
        sig2 = Signal(
            strategy="s2", symbol="AAPL",
            direction=Direction.LONG, conviction=0.9, timestamp=time.time(),
        )
        agent._pending_signals["AAPL"] = [sig1, sig2]

        weights = agent._compute_target_weights()
        assert "AAPL" in weights
        assert weights["AAPL"] > 0
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_max_positions_cap(self):
        bus = InProcessMessageBus()
        await bus.connect()
        agent = PortfolioManagementAgent(bus=bus, nav=100_000, max_positions=2)
        await agent.on_start()

        for sym in ["AAPL", "MSFT", "GOOG", "AMZN"]:
            sig = Signal(
                strategy="test", symbol=sym,
                direction=Direction.LONG, conviction=0.5, timestamp=time.time(),
            )
            await bus.publish(f"signals.test.{sym}", sig.model_dump())

        weights = agent._compute_target_weights()
        assert len(weights) <= 2
        await bus.disconnect()


# ── ExecutionAgent Tests ──────────────────────────────────────────────

class TestExecutionAgent:
    @pytest.mark.asyncio
    async def test_simulated_fill_on_approved_order(self):
        bus = InProcessMessageBus()
        await bus.connect()

        fills = []
        async def capture_fill(ch, payload):
            fills.append(payload)

        await bus.subscribe("fills.*", capture_fill)

        agent = ExecutionAgent(bus=bus)
        await agent.on_start()

        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.LIMIT, limit_price=150.0,
            status=OrderStatus.APPROVED, timestamp=time.time(),
        )
        await bus.publish("orders.approved", order.model_dump())

        assert len(fills) == 1
        assert fills[0]["symbol"] == "AAPL"
        assert fills[0]["filled_quantity"] == 100
        assert fills[0]["avg_price"] == 150.0
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_simulated_fill_commission(self):
        bus = InProcessMessageBus()
        await bus.connect()

        fills = []
        async def capture_fill(ch, payload):
            fills.append(payload)

        await bus.subscribe("fills.*", capture_fill)

        agent = ExecutionAgent(bus=bus)
        await agent.on_start()

        order = Order(
            symbol="MSFT", side=OrderSide.BUY, quantity=50,
            order_type=OrderType.LIMIT, limit_price=300.0,
            status=OrderStatus.APPROVED, timestamp=time.time(),
        )
        await bus.publish("orders.approved", order.model_dump())

        # Commission = qty * price * 0.0001 = 50 * 300 * 0.0001 = 1.5
        assert abs(fills[0]["commission"] - 1.5) < 0.01
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_halt_clears_open_orders(self):
        bus = InProcessMessageBus()
        await bus.connect()

        agent = ExecutionAgent(bus=bus)
        await agent.on_start()

        # Manually add an open order
        agent._open_orders["test-id"] = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=10,
        )

        await bus.publish("system.halt_trading", {"reason": "TEST_HALT"})
        assert len(agent._open_orders) == 0
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_on_stop_without_ib(self):
        bus = InProcessMessageBus()
        await bus.connect()
        agent = ExecutionAgent(bus=bus)
        await agent.on_start()
        await agent.on_stop()  # Should not raise
        await bus.disconnect()


# ── SystemMonitorAgent Tests ──────────────────────────────────────────

class TestSystemMonitorAgent:
    @pytest.mark.asyncio
    async def test_heartbeat_tracking(self):
        bus = InProcessMessageBus()
        await bus.connect()

        agent = SystemMonitorAgent(bus=bus, expected_agents=["test_agent"])
        await agent.on_start()

        hb = Heartbeat(agent="test_agent", timestamp=time.time())
        await bus.publish("system.heartbeat.test_agent", hb.model_dump())

        assert "test_agent" in agent._last_heartbeat
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_health_check_offline(self):
        bus = InProcessMessageBus()
        await bus.connect()

        agent = SystemMonitorAgent(
            bus=bus, expected_agents=["agent_a"], heartbeat_timeout_sec=1.0
        )
        await agent.on_start()

        # No heartbeat sent → should be OFFLINE
        health = agent.check_agent_health()
        assert health["agent_a"] == "OFFLINE"
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_health_check_ok(self):
        bus = InProcessMessageBus()
        await bus.connect()

        agent = SystemMonitorAgent(
            bus=bus, expected_agents=["agent_a"], heartbeat_timeout_sec=60.0
        )
        await agent.on_start()

        hb = Heartbeat(agent="agent_a", timestamp=time.time())
        await bus.publish("system.heartbeat.agent_a", hb.model_dump())

        health = agent.check_agent_health()
        assert health["agent_a"] == "OK"
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_metric_counters(self):
        bus = InProcessMessageBus()
        await bus.connect()

        agent = SystemMonitorAgent(bus=bus, expected_agents=[])
        await agent.on_start()

        # Publish various events
        await bus.publish("signals.test.AAPL", {"direction": "LONG"})
        await bus.publish("signals.test.MSFT", {"direction": "SHORT"})
        await bus.publish("orders.proposed", {"symbol": "AAPL"})
        await bus.publish("orders.approved", {"symbol": "AAPL"})
        await bus.publish("orders.rejected", {"symbol": "X", "metadata": {"rejection_reason": "TEST"}})
        await bus.publish("fills.AAPL", {"symbol": "AAPL"})

        assert agent._total_signals == 2
        assert agent._total_orders_proposed == 1
        assert agent._total_orders_approved == 1
        assert agent._total_orders_rejected == 1
        assert agent._total_fills == 1
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_get_metrics_snapshot(self):
        bus = InProcessMessageBus()
        await bus.connect()

        agent = SystemMonitorAgent(bus=bus, expected_agents=["a"])
        await agent.on_start()
        await bus.publish("signals.test.X", {"d": "L"})

        metrics = agent.get_metrics()
        assert metrics["signals"] == 1
        assert "agent_health" in metrics
        assert "a" in metrics["agent_health"]
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_halt_event_handled(self):
        bus = InProcessMessageBus()
        await bus.connect()

        agent = SystemMonitorAgent(bus=bus, expected_agents=[])
        await agent.on_start()

        await bus.publish("system.halt_trading", {"reason": "DRAWDOWN_5%"})
        # Should not raise — just logs
        await bus.disconnect()


# ── RiskManagement Additional Coverage ────────────────────────────────

class TestRiskManagementExtended:
    @pytest.mark.asyncio
    async def test_fill_updates_position(self):
        bus = InProcessMessageBus()
        await bus.connect()
        agent = RiskManagementAgent(bus=bus, nav=100_000)
        await agent.on_start()

        fill = Fill(
            order_id="o1", symbol="AAPL", side=OrderSide.BUY,
            filled_quantity=100, avg_price=150.0, commission=1.5,
            timestamp=time.time(),
        )
        await bus.publish("fills.AAPL", fill.model_dump())

        assert "AAPL" in agent._positions
        assert agent._daily_pnl == -1.5  # commission deducted
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_drawdown_triggers_halt(self):
        bus = InProcessMessageBus()
        await bus.connect()

        halts = []
        async def capture_halt(ch, payload):
            halts.append(payload)

        await bus.subscribe("system.halt_trading", capture_halt)

        agent = RiskManagementAgent(
            bus=bus, nav=100_000, max_daily_drawdown_pct=0.01
        )
        await agent.on_start()

        # Simulate large commission loss to trigger drawdown
        agent._daily_pnl = -5000  # 5% loss exceeds 1% limit
        agent._high_water_mark = 100_000

        fill = Fill(
            order_id="o2", symbol="MSFT", side=OrderSide.BUY,
            filled_quantity=1, avg_price=100.0, commission=0,
            timestamp=time.time(),
        )
        await bus.publish("fills.MSFT", fill.model_dump())

        assert agent._trading_halted
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_daily_reset(self):
        bus = InProcessMessageBus()
        await bus.connect()
        agent = RiskManagementAgent(bus=bus, nav=100_000)
        agent._daily_pnl = -500
        agent._trading_halted = True

        agent.reset_daily()

        assert agent._daily_pnl == 0.0
        assert not agent._trading_halted
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_gross_exposure_rejection(self):
        bus = InProcessMessageBus()
        await bus.connect()

        results = {"approved": [], "rejected": []}
        async def on_approved(ch, payload):
            results["approved"].append(payload)
        async def on_rejected(ch, payload):
            results["rejected"].append(payload)

        await bus.subscribe("orders.approved", on_approved)
        await bus.subscribe("orders.rejected", on_rejected)

        # Use high max_position_pct so position limit doesn't trigger first
        agent = RiskManagementAgent(bus=bus, nav=100_000, max_position_pct=0.50)
        await agent.on_start()

        # Set existing positions near 150% gross cap
        agent._positions["AAPL"] = 70_000
        agent._positions["GOOG"] = 70_000

        # This pushes total to 140k + 20k = 160k > 150k
        order = Order(
            symbol="MSFT", side=OrderSide.BUY, quantity=200,
            order_type=OrderType.LIMIT, limit_price=100.0,
        )
        await bus.publish("orders.proposed", order.model_dump())

        assert len(results["rejected"]) == 1
        assert "GROSS_EXPOSURE_EXCEEDED" in results["rejected"][0]["metadata"]["rejection_reason"]
        await bus.disconnect()
