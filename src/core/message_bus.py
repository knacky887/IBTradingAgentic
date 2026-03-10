"""
TradingApp — Message Bus

Dual-mode message bus:
  • LIVE  → Redis Pub/Sub + Streams (real network I/O)
  • BACKTEST → in-process asyncio.Queue (deterministic, zero-latency)

Both modes expose the same async interface so agents are mode-agnostic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine

import msgpack

logger = logging.getLogger("tradingapp.core.message_bus")

# Type alias for subscription callbacks
MessageCallback = Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]]


# ── Abstract Interface ─────────────────────────────────────────────────

class MessageBus(ABC):
    """Abstract message bus interface."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the underlying transport."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully shut down."""

    @abstractmethod
    async def publish(self, channel: str, payload: dict[str, Any]) -> None:
        """Publish a message to a channel."""

    @abstractmethod
    async def subscribe(self, pattern: str, callback: MessageCallback) -> None:
        """
        Subscribe to channels matching *pattern* (supports glob wildcards).
        The callback receives (channel: str, payload: dict).
        """

    @abstractmethod
    async def stream_append(self, stream: str, payload: dict[str, Any]) -> str:
        """Append a message to a Redis Stream (or equivalent). Returns message ID."""

    @abstractmethod
    async def stream_read(
        self, stream: str, last_id: str = "0-0", count: int = 100
    ) -> list[tuple[str, dict[str, Any]]]:
        """Read messages from a stream after *last_id*."""


# ── Redis Implementation ───────────────────────────────────────────────

class RedisMessageBus(MessageBus):
    """Production message bus backed by Redis Pub/Sub + Streams."""

    def __init__(self, host: str = "127.0.0.1", port: int = 6379, db: int = 0, password: str = ""):
        self._host = host
        self._port = port
        self._db = db
        self._password = password
        self._redis = None              # redis.asyncio.Redis
        self._pubsub = None             # redis.asyncio.PubSub
        self._listener_task: asyncio.Task | None = None
        self._callbacks: dict[str, list[MessageCallback]] = {}
        self._running = False

    async def connect(self) -> None:
        import redis.asyncio as aioredis

        self._redis = aioredis.Redis(
            host=self._host,
            port=self._port,
            db=self._db,
            password=self._password or None,
            decode_responses=False,
        )
        self._pubsub = self._redis.pubsub()
        self._running = True
        logger.info("Redis message bus connected to %s:%s", self._host, self._port)

    async def disconnect(self) -> None:
        self._running = False
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        if self._pubsub:
            await self._pubsub.punsubscribe()
            await self._pubsub.close()
        if self._redis:
            await self._redis.close()
        logger.info("Redis message bus disconnected.")

    async def publish(self, channel: str, payload: dict[str, Any]) -> None:
        data = msgpack.packb(payload, use_bin_type=True)
        await self._redis.publish(channel, data)

    async def subscribe(self, pattern: str, callback: MessageCallback) -> None:
        self._callbacks.setdefault(pattern, []).append(callback)
        await self._pubsub.psubscribe(pattern)
        # Start listener if not already running
        if self._listener_task is None or self._listener_task.done():
            self._listener_task = asyncio.create_task(self._listen())

    async def _listen(self) -> None:
        """Background task that listens for Pub/Sub messages."""
        try:
            while self._running:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message is None:
                    continue
                channel = (
                    message["channel"].decode("utf-8")
                    if isinstance(message["channel"], bytes)
                    else message["channel"]
                )
                payload = msgpack.unpackb(message["data"], raw=False)
                pattern = (
                    message.get("pattern", b"").decode("utf-8")
                    if isinstance(message.get("pattern", b""), bytes)
                    else message.get("pattern", "")
                )
                for cb in self._callbacks.get(pattern, []):
                    try:
                        await cb(channel, payload)
                    except Exception:
                        logger.exception("Callback error on channel %s", channel)
        except asyncio.CancelledError:
            pass

    async def stream_append(self, stream: str, payload: dict[str, Any]) -> str:
        data = {b"data": msgpack.packb(payload, use_bin_type=True)}
        msg_id = await self._redis.xadd(stream, data)
        return msg_id.decode("utf-8") if isinstance(msg_id, bytes) else msg_id

    async def stream_read(
        self, stream: str, last_id: str = "0-0", count: int = 100
    ) -> list[tuple[str, dict[str, Any]]]:
        results = await self._redis.xread({stream: last_id}, count=count, block=100)
        messages = []
        for _stream_name, entries in results:
            for msg_id, fields in entries:
                mid = msg_id.decode("utf-8") if isinstance(msg_id, bytes) else msg_id
                payload = msgpack.unpackb(fields[b"data"], raw=False)
                messages.append((mid, payload))
        return messages


# ── In-Process Implementation (Backtest) ───────────────────────────────

class InProcessMessageBus(MessageBus):
    """
    Zero-latency, deterministic message bus for backtesting.
    Uses asyncio.Queue under the hood — no network I/O.
    """

    def __init__(self):
        self._callbacks: dict[str, list[MessageCallback]] = {}
        self._streams: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self._stream_counter: int = 0

    async def connect(self) -> None:
        logger.info("In-process message bus ready (backtest mode).")

    async def disconnect(self) -> None:
        self._callbacks.clear()
        self._streams.clear()
        logger.info("In-process message bus shut down.")

    async def publish(self, channel: str, payload: dict[str, Any]) -> None:
        """Immediately dispatch to all matching callbacks (deterministic)."""
        for pattern, cbs in list(self._callbacks.items()):
            if self._pattern_matches(pattern, channel):
                for cb in list(cbs):
                    try:
                        await cb(channel, payload)
                    except Exception:
                        logger.exception("Callback error on channel %s", channel)

    async def subscribe(self, pattern: str, callback: MessageCallback) -> None:
        self._callbacks.setdefault(pattern, []).append(callback)

    async def stream_append(self, stream: str, payload: dict[str, Any]) -> str:
        self._stream_counter += 1
        msg_id = f"{int(time.time() * 1000)}-{self._stream_counter}"
        self._streams.setdefault(stream, []).append((msg_id, payload))
        return msg_id

    async def stream_read(
        self, stream: str, last_id: str = "0-0", count: int = 100
    ) -> list[tuple[str, dict[str, Any]]]:
        entries = self._streams.get(stream, [])
        # Simple filter: return entries after last_id
        result = []
        found = last_id == "0-0"
        for mid, payload in entries:
            if found:
                result.append((mid, payload))
                if len(result) >= count:
                    break
            elif mid == last_id:
                found = True
        return result

    @staticmethod
    def _pattern_matches(pattern: str, channel: str) -> bool:
        """
        Glob matching for dot-delimited channel names.
        Supports '*' as single-level wildcard and '**' as multi-level wildcard.
        """
        p_parts = pattern.split(".")
        c_parts = channel.split(".")
        return InProcessMessageBus._glob_match(p_parts, c_parts, 0, 0)

    @staticmethod
    def _glob_match(p: list[str], c: list[str], pi: int, ci: int) -> bool:
        """Recursive glob matcher supporting * (single segment) and ** (multi-level)."""
        while pi < len(p) and ci < len(c):
            if p[pi] == "**":
                # ** matches zero or more segments
                for k in range(len(c) - ci + 1):
                    if InProcessMessageBus._glob_match(p, c, pi + 1, ci + k):
                        return True
                return False
            elif p[pi] == "*" or p[pi] == c[ci]:
                pi += 1
                ci += 1
            else:
                return False
        # Skip trailing ** patterns that match zero segments
        while pi < len(p) and p[pi] == "**":
            pi += 1
        return pi == len(p) and ci == len(c)
