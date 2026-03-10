"""
TradingApp — Base Agent

Abstract base class that all agents inherit. Provides:
  • Lifecycle management (start → run → stop)
  • Automatic heartbeat publishing
  • Message bus integration
  • Structured logging
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod

from src.core.message_bus import MessageBus
from src.core.models import Heartbeat

logger = logging.getLogger("tradingapp.core.base_agent")


class BaseAgent(ABC):
    """
    Abstract base for every agent in the system.

    Subclasses implement:
      • on_start()   — subscribe to channels, initialise state
      • on_message() — handle incoming messages (via subscriptions)
      • on_stop()    — cleanup
    """

    def __init__(
        self,
        name: str,
        bus: MessageBus,
        heartbeat_interval: float = 10.0,
    ):
        self.name = name
        self.bus = bus
        self._heartbeat_interval = heartbeat_interval
        self._running = False
        self._heartbeat_task: asyncio.Task | None = None
        self.logger = logging.getLogger(f"tradingapp.agents.{name}")

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect to the bus, run on_start(), begin heartbeat."""
        self.logger.info("Agent '%s' starting…", self.name)
        self._running = True

        await self.on_start()

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.logger.info("Agent '%s' is running.", self.name)

    async def stop(self) -> None:
        """Stop heartbeat, run on_stop(), clean up."""
        self.logger.info("Agent '%s' stopping…", self.name)
        self._running = False

        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        await self.on_stop()
        self.logger.info("Agent '%s' stopped.", self.name)

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Abstract hooks ─────────────────────────────────────────────────

    @abstractmethod
    async def on_start(self) -> None:
        """
        Called once at startup.
        Use this to subscribe to channels and initialise agent state.
        """

    async def on_stop(self) -> None:
        """Called once at shutdown. Override to clean up resources."""

    # ── Helpers ────────────────────────────────────────────────────────

    async def publish(self, channel: str, payload: dict) -> None:
        """Convenience wrapper for bus.publish with source tagging."""
        envelope = {**payload, "_source": self.name, "_timestamp": time.time()}
        await self.bus.publish(channel, envelope)

    async def subscribe(self, pattern: str, callback) -> None:
        """Convenience wrapper for bus.subscribe."""
        await self.bus.subscribe(pattern, callback)

    # ── Heartbeat ──────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """Publish heartbeat on a fixed interval."""
        try:
            while self._running:
                hb = Heartbeat(
                    agent=self.name,
                    timestamp=time.time(),
                    status="OK",
                )
                await self.bus.publish(
                    f"system.heartbeat.{self.name}",
                    hb.model_dump(),
                )
                await asyncio.sleep(self._heartbeat_interval)
        except asyncio.CancelledError:
            pass
