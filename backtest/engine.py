"""
TradingApp — Event-Driven Backtest Engine

Replays historical bars through the in-process message bus,
runs agents in deterministic order, and records an equity curve.
Prevents lookahead bias by gating bar access.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.core.message_bus import InProcessMessageBus
from src.core.models import Bar, Fill, OrderSide
from src.agents.alpha_generation import AlphaGenerationAgent
from src.agents.risk_management import RiskManagementAgent
from src.agents.portfolio_management import PortfolioManagementAgent
from src.agents.execution import ExecutionAgent
from src.utils.data_store import load_parquet, bars_to_dataframe

logger = logging.getLogger("tradingapp.backtest.engine")


class BacktestEngine:
    """
    Event-driven backtest engine.

    Replays bars one at a time through the in-process message bus.
    All agents run identical code as in live mode.
    """

    def __init__(
        self,
        strategy_config: dict[str, Any],
        initial_nav: float = 1_000_000.0,
    ):
        self.bus = InProcessMessageBus()
        self.strategy_config = strategy_config
        self.initial_nav = initial_nav

        # Agents
        self.alpha_agent = AlphaGenerationAgent(bus=self.bus, strategy_config=strategy_config)
        self.risk_agent = RiskManagementAgent(bus=self.bus, nav=initial_nav)
        self.portfolio_agent = PortfolioManagementAgent(bus=self.bus, nav=initial_nav)
        self.execution_agent = ExecutionAgent(bus=self.bus)

        # Results
        self._equity_curve: list[dict[str, Any]] = []
        self._fills: list[Fill] = []
        self._nav = initial_nav

    async def run(self, bars: list[Bar]) -> pd.DataFrame:
        """
        Run backtest over the provided bars.

        Returns:
            DataFrame with columns: timestamp, nav, returns
        """
        logger.info("Starting backtest with %d bars, NAV=%.0f", len(bars), self.initial_nav)

        # Connect bus and start agents
        await self.bus.connect()
        await self.alpha_agent.on_start()
        await self.risk_agent.on_start()
        await self.portfolio_agent.on_start()
        await self.execution_agent.on_start()

        # Subscribe to fills for equity tracking
        await self.bus.subscribe("fills.*", self._on_fill)

        # Replay bars one by one (prevents lookahead)
        for i, bar in enumerate(bars):
            # Publish bar — this triggers alpha → portfolio → risk → execution chain
            await self.bus.publish(f"market.bars.{bar.symbol}", bar.model_dump())

            # Trigger rebalance periodically (every 60 bars ≈ hourly on 1-min bars)
            if (i + 1) % 60 == 0:
                await self.portfolio_agent.rebalance()

            # Record equity
            self._equity_curve.append({
                "timestamp": bar.timestamp,
                "nav": self._nav,
                "bar_index": i,
            })

            if (i + 1) % 1000 == 0:
                logger.info("Backtest progress: %d / %d bars", i + 1, len(bars))

        # Final rebalance
        await self.portfolio_agent.rebalance()

        # Disconnect
        await self.bus.disconnect()

        # Build results
        df = pd.DataFrame(self._equity_curve)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            df = df.set_index("datetime")
            df["returns"] = df["nav"].pct_change().fillna(0)

        logger.info(
            "Backtest complete: %d bars, %d fills, final NAV=%.0f",
            len(bars), len(self._fills), self._nav,
        )
        return df

    async def _on_fill(self, channel: str, payload: dict[str, Any]) -> None:
        """Track simulated fills for NAV computation."""
        fill = Fill(**payload)
        self._fills.append(fill)

        # Simplified PnL: commission + slippage deducted from NAV
        self._nav -= fill.commission

    def compute_stats(self, equity_df: pd.DataFrame) -> dict[str, float]:
        """Compute performance statistics from equity curve."""
        if equity_df.empty or "returns" not in equity_df.columns:
            return {}

        returns = equity_df["returns"].dropna()

        total_return = (equity_df["nav"].iloc[-1] / equity_df["nav"].iloc[0]) - 1
        annual_factor = 252 * 390  # ~1-min bars per year

        # Annualised return
        n_bars = len(returns)
        ann_return = (1 + total_return) ** (annual_factor / max(n_bars, 1)) - 1

        # Volatility
        ann_vol = returns.std() * np.sqrt(annual_factor)

        # Sharpe
        sharpe = ann_return / max(ann_vol, 1e-10)

        # Max drawdown
        cummax = equity_df["nav"].cummax()
        drawdown = (equity_df["nav"] - cummax) / cummax
        max_dd = drawdown.min()

        return {
            "total_return": total_return,
            "annualised_return": ann_return,
            "annualised_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "total_fills": len(self._fills),
            "final_nav": self._nav,
        }
