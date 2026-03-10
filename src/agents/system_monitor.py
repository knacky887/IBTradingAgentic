"""
TradingApp — System Monitor Agent

Health checking, performance metrics, structured logging, and alerting.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.core.base_agent import BaseAgent
from src.core.message_bus import MessageBus
from src.core.models import Heartbeat

logger = logging.getLogger("tradingapp.agents.system_monitor")


class SystemMonitorAgent(BaseAgent):
    """
    Responsible for:
      1. Heartbeat monitoring — escalate if agents miss beats
      2. Performance metrics aggregation (PnL, Sharpe)
      3. Structured logging of all system events
      4. Alerting on critical events
    """

    def __init__(
        self,
        bus: MessageBus,
        expected_agents: list[str] | None = None,
        heartbeat_timeout_sec: float = 30.0,
    ):
        super().__init__(name="system_monitor", bus=bus)
        self._expected_agents = expected_agents or [
            "data_ingestion",
            "alpha_generation",
            "risk_management",
            "portfolio_management",
            "execution",
        ]
        self._heartbeat_timeout = heartbeat_timeout_sec
        self._last_heartbeat: dict[str, float] = {}

        # Metrics
        self._total_signals: int = 0
        self._total_orders_proposed: int = 0
        self._total_orders_approved: int = 0
        self._total_orders_rejected: int = 0
        self._total_fills: int = 0

    async def on_start(self) -> None:
        await self.subscribe("system.heartbeat.*", self._on_heartbeat)
        await self.subscribe("signals.*.*", self._on_signal)
        await self.subscribe("orders.proposed", self._on_order_proposed)
        await self.subscribe("orders.approved", self._on_order_approved)
        await self.subscribe("orders.rejected", self._on_order_rejected)
        await self.subscribe("fills.*", self._on_fill)
        await self.subscribe("system.halt_trading", self._on_halt)
        self.logger.info("System Monitor started — watching %d agents", len(self._expected_agents))

    # ── Heartbeat ──────────────────────────────────────────────────────

    async def _on_heartbeat(self, channel: str, payload: dict[str, Any]) -> None:
        hb = Heartbeat(**payload)
        self._last_heartbeat[hb.agent] = time.time()

    def check_agent_health(self) -> dict[str, str]:
        """Check which agents have missed heartbeats."""
        now = time.time()
        status = {}
        for agent in self._expected_agents:
            last = self._last_heartbeat.get(agent, 0)
            if now - last > self._heartbeat_timeout:
                status[agent] = "OFFLINE"
                self.logger.error("Agent '%s' appears OFFLINE (no heartbeat for %.0fs)", agent, now - last)
            else:
                status[agent] = "OK"
        return status

    # ── Metric Counters ────────────────────────────────────────────────

    async def _on_signal(self, channel: str, payload: dict[str, Any]) -> None:
        self._total_signals += 1

    async def _on_order_proposed(self, channel: str, payload: dict[str, Any]) -> None:
        self._total_orders_proposed += 1

    async def _on_order_approved(self, channel: str, payload: dict[str, Any]) -> None:
        self._total_orders_approved += 1

    async def _on_order_rejected(self, channel: str, payload: dict[str, Any]) -> None:
        self._total_orders_rejected += 1
        self.logger.warning("Order rejected: %s", payload.get("metadata", {}).get("rejection_reason", "unknown"))

    async def _on_fill(self, channel: str, payload: dict[str, Any]) -> None:
        self._total_fills += 1

    async def _on_halt(self, channel: str, payload: dict[str, Any]) -> None:
        self.logger.critical("SYSTEM HALT: %s", payload.get("reason", "unknown"))
        # TODO: Send Slack/email alert

    # ── Metrics Snapshot ───────────────────────────────────────────────

    def get_metrics(self) -> dict[str, Any]:
        return {
            "signals": self._total_signals,
            "orders_proposed": self._total_orders_proposed,
            "orders_approved": self._total_orders_approved,
            "orders_rejected": self._total_orders_rejected,
            "fills": self._total_fills,
            "agent_health": self.check_agent_health(),
        }
