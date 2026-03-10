"""
TradingApp — Data Ingestion Agent

Connects to Interactive Brokers (or a simulated source in backtest mode),
normalises market data into canonical Bar schema, publishes to the message bus,
and persists to Redis TimeSeries + Parquet.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from src.core.base_agent import BaseAgent
from src.core.message_bus import MessageBus
from src.core.models import Bar

logger = logging.getLogger("tradingapp.agents.data_ingestion")


class DataIngestionAgent(BaseAgent):
    """
    Responsible for:
      1. Connecting to IB Gateway and subscribing to real-time bars
      2. Normalising data into the canonical Bar schema
      3. Publishing bars on  market.bars.{symbol}
      4. Storing bars in Redis TimeSeries for lookback
      5. Periodically flushing to Parquet for long-term storage
    """

    def __init__(
        self,
        bus: MessageBus,
        symbols: list[str],
        bar_size: str = "1 min",
        ib_host: str = "127.0.0.1",
        ib_port: int = 7497,
        ib_client_id: int = 1,
    ):
        super().__init__(name="data_ingestion", bus=bus)
        self.symbols = symbols
        self.bar_size = bar_size
        self._ib_host = ib_host
        self._ib_port = ib_port
        self._ib_client_id = ib_client_id
        self._ib = None          # ib_insync.IB instance
        self._bar_buffer: list[Bar] = []

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def on_start(self) -> None:
        """Connect to IB and subscribe to real-time bars."""
        self.logger.info(
            "Data Ingestion starting — symbols=%s, bar_size=%s",
            self.symbols, self.bar_size,
        )
        try:
            await self._connect_ib()
            await self._subscribe_bars()
        except Exception:
            self.logger.exception("Failed to connect to IB — running in degraded mode")

    async def on_stop(self) -> None:
        """Disconnect from IB."""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            self.logger.info("Disconnected from IB Gateway.")

    # ── IB Connection ──────────────────────────────────────────────────

    async def _connect_ib(self) -> None:
        """Connect to IB Gateway using ib_insync."""
        from ib_insync import IB
        self._ib = IB()
        await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self._ib.connect(
                self._ib_host, self._ib_port, clientId=self._ib_client_id
            ),
        )
        self.logger.info("Connected to IB Gateway at %s:%s", self._ib_host, self._ib_port)

    async def _subscribe_bars(self) -> None:
        """Subscribe to real-time bar updates for each symbol."""
        from ib_insync import Stock

        for symbol in self.symbols:
            contract = Stock(symbol, "SMART", "USD")
            await asyncio.get_running_loop().run_in_executor(
                None, lambda c=contract: self._ib.qualifyContracts(c)
            )
            bars = self._ib.reqRealTimeBars(
                contract, barSize=5, whatToShow="MIDPOINT", useRTH=True
            )
            bars.updateEvent += lambda b, s=symbol: asyncio.create_task(
                self._on_bar_update(s, b)
            )
            self.logger.info("Subscribed to real-time bars for %s", symbol)

    # ── Bar Handling ───────────────────────────────────────────────────

    async def _on_bar_update(self, symbol: str, bars: Any) -> None:
        """Called whenever a new real-time bar arrives from IB."""
        if not bars:
            return
        latest = bars[-1]
        bar = Bar(
            symbol=symbol,
            timestamp=latest.time.timestamp() if hasattr(latest.time, "timestamp") else time.time(),
            open=float(getattr(latest, "open_", getattr(latest, "open", 0))),
            high=float(latest.high),
            low=float(latest.low),
            close=float(latest.close),
            volume=float(getattr(latest, "volume", 0)),
            vwap=float(getattr(latest, "wap", 0)) or None,
        )
        # Publish to bus
        await self.publish(f"market.bars.{symbol}", bar.model_dump())
        self._bar_buffer.append(bar)
        self.logger.debug("Published bar: %s @ %.2f", symbol, bar.close)

    # ── Simulated Bar Injection (Backtest) ─────────────────────────────

    async def inject_bar(self, bar: Bar) -> None:
        """
        Used by the backtest engine to inject bars directly
        without an IB connection.
        """
        await self.publish(f"market.bars.{bar.symbol}", bar.model_dump())
