"""
TradingApp — Unit Tests for Core Models and Message Bus
"""

import asyncio
import pytest
from src.core.models import Bar, Signal, Order, Fill, Message, Direction, OrderSide, OrderType, OrderStatus
from src.core.message_bus import InProcessMessageBus


# ── Model Serialization Tests ──────────────────────────────────────────

class TestBarModel:
    def test_bar_creation(self):
        bar = Bar(
            symbol="AAPL", timestamp=1700000000.0,
            open=150.0, high=152.0, low=149.0, close=151.0, volume=1e6,
        )
        assert bar.symbol == "AAPL"
        assert bar.close == 151.0

    def test_bar_roundtrip(self):
        bar = Bar(
            symbol="MSFT", timestamp=1700000000.0,
            open=300.0, high=305.0, low=298.0, close=303.0, volume=5e5, vwap=302.0,
        )
        data = bar.model_dump()
        restored = Bar(**data)
        assert restored == bar


class TestSignalModel:
    def test_signal_creation(self):
        sig = Signal(
            strategy="dual_ma", symbol="AAPL",
            direction=Direction.LONG, conviction=0.8, timestamp=1700000000.0,
        )
        assert sig.direction == Direction.LONG
        assert sig.conviction == 0.8
        assert len(sig.id) > 0  # UUID generated


class TestOrderModel:
    def test_order_defaults(self):
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        assert order.status == OrderStatus.PROPOSED
        assert order.order_type == OrderType.LIMIT


class TestFillModel:
    def test_fill_creation(self):
        fill = Fill(
            order_id="test-id", symbol="AAPL", side=OrderSide.BUY,
            filled_quantity=100, avg_price=150.0, commission=1.0,
        )
        assert fill.filled_quantity == 100


class TestMessageEnvelope:
    def test_envelope_roundtrip(self):
        msg = Message(
            timestamp=1700000000.0, channel="market.bars.AAPL",
            source="data_ingestion", payload={"close": 151.0},
        )
        data = msg.model_dump()
        restored = Message(**data)
        assert restored.channel == msg.channel


# ── Message Bus Tests ──────────────────────────────────────────────────

class TestInProcessMessageBus:
    @pytest.fixture
    def bus(self):
        return InProcessMessageBus()

    @pytest.mark.asyncio
    async def test_publish_subscribe(self, bus):
        received = []

        async def handler(channel, payload):
            received.append((channel, payload))

        await bus.connect()
        await bus.subscribe("test.channel", handler)
        await bus.publish("test.channel", {"value": 42})

        assert len(received) == 1
        assert received[0][0] == "test.channel"
        assert received[0][1]["value"] == 42
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_wildcard_subscribe(self, bus):
        received = []

        async def handler(channel, payload):
            received.append(channel)

        await bus.connect()
        await bus.subscribe("market.bars.*", handler)
        await bus.publish("market.bars.AAPL", {"close": 150})
        await bus.publish("market.bars.MSFT", {"close": 300})
        await bus.publish("signals.test.AAPL", {"direction": "LONG"})  # should NOT match

        assert len(received) == 2
        assert "market.bars.AAPL" in received
        assert "market.bars.MSFT" in received
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_stream_append_read(self, bus):
        await bus.connect()
        mid = await bus.stream_append("test_stream", {"value": 1})
        assert mid is not None

        msgs = await bus.stream_read("test_stream")
        assert len(msgs) == 1
        assert msgs[0][1]["value"] == 1
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_no_match(self, bus):
        received = []

        async def handler(channel, payload):
            received.append(channel)

        await bus.connect()
        await bus.subscribe("market.bars.*", handler)
        await bus.publish("signals.test.AAPL", {"data": 1})

        assert len(received) == 0
        await bus.disconnect()
