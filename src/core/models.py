"""
TradingApp — Canonical Data Models

Pydantic models for every message type flowing through the system.
All timestamps are Unix epoch floats in UTC.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────

class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MKT"
    LIMIT = "LIMIT"
    ADAPTIVE = "ADAPTIVE"


class OrderStatus(str, Enum):
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


# ── Market Data ────────────────────────────────────────────────────────

class Bar(BaseModel):
    """Canonical OHLCV bar."""
    symbol: str
    timestamp: float               # Unix epoch UTC
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None

    model_config = {"frozen": True}


# ── Signals ────────────────────────────────────────────────────────────

class Signal(BaseModel):
    """Output of a strategy — a directional conviction on a symbol."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    strategy: str
    symbol: str
    direction: Direction
    conviction: float              # 0.0 – 1.0
    timestamp: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Orders ─────────────────────────────────────────────────────────────

class Order(BaseModel):
    """A proposed, approved, or rejected order."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.LIMIT
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PROPOSED
    strategy: str = ""
    signal_id: str = ""
    timestamp: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Fills ──────────────────────────────────────────────────────────────

class Fill(BaseModel):
    """Execution fill confirmation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str
    symbol: str
    side: OrderSide
    filled_quantity: int
    avg_price: float
    commission: float = 0.0
    slippage_bps: float = 0.0
    timestamp: float = 0.0


# ── Message Envelope ───────────────────────────────────────────────────

class Message(BaseModel):
    """
    Canonical message envelope wrapping any payload.
    Every message on the bus is wrapped in this envelope.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float
    channel: str
    source: str                    # agent name
    payload: Dict[str, Any]
    version: int = 1


# ── Heartbeat ──────────────────────────────────────────────────────────

class Heartbeat(BaseModel):
    agent: str
    timestamp: float
    status: str = "OK"             # OK | DEGRADED | ERROR
    metadata: Dict[str, Any] = Field(default_factory=dict)
