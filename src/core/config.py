"""
TradingApp — Configuration Loader

Loads and validates YAML configuration files with environment variable overrides.
"""

import os
import logging
import logging.config
from pathlib import Path
from typing import Any, List, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger("tradingapp.core.config")

# ── Locate config directory ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


# ── Pydantic Settings Models ──────────────────────────────────────────

class BrokerSettings(BaseModel):
    name: str = "interactive_brokers"
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    account: str = ""


class RedisSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 6379
    db: int = 0
    password: str = ""


class UniverseSettings(BaseModel):
    source: str = "sp500"
    custom_symbols: List[str] = Field(default_factory=list)
    refresh_interval_days: int = 7


class RiskSettings(BaseModel):
    max_position_pct: float = 0.05
    max_sector_pct: float = 0.25
    max_daily_drawdown_pct: float = 0.02
    max_correlated_positions: int = 5
    adv_min_multiplier: float = 0.01
    var_confidence: float = 0.99
    var_lookback_days: int = 252


class PortfolioSettings(BaseModel):
    rebalance_cadence: str = "EOD"
    signal_combination: str = "conviction"
    max_positions: int = 50


class ExecutionSettings(BaseModel):
    default_order_type: str = "LIMIT"
    twap_slice_minutes: int = 5
    max_slippage_bps: int = 10
    allow_extended_hours: bool = False


class DataSettings(BaseModel):
    bar_size: str = "1 min"
    history_days: int = 504
    storage_path: str = "data/parquet"
    redis_timeseries_retention_ms: int = 604_800_000


class SystemSettings(BaseModel):
    mode: str = "paper"
    log_level: str = "INFO"
    heartbeat_interval_sec: int = 10
    timezone: str = "US/Eastern"


class AppSettings(BaseModel):
    """Top-level settings container."""
    system: SystemSettings = Field(default_factory=SystemSettings)
    broker: BrokerSettings = Field(default_factory=BrokerSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    universe: UniverseSettings = Field(default_factory=UniverseSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    portfolio: PortfolioSettings = Field(default_factory=PortfolioSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    data: DataSettings = Field(default_factory=DataSettings)


# ── Loaders ────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Override config values with environment variables.
    Convention: TRADINGAPP_<SECTION>_<KEY> in uppercase.
    Example:  TRADINGAPP_BROKER_PORT=4002
    """
    prefix = "TRADINGAPP_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("_", 1)
        if len(parts) == 2:
            section, param = parts
            if section in raw and isinstance(raw[section], dict):
                # Attempt numeric coercion
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                raw[section][param] = value
                logger.info("Env override: %s.%s = %s", section, param, value)
    return raw


def load_settings(path: Optional[Path] = None) -> AppSettings:
    """Load and validate application settings."""
    path = path or CONFIG_DIR / "settings.yaml"
    raw = _load_yaml(path)
    raw = _apply_env_overrides(raw)
    return AppSettings(**raw)


def load_strategy_config(path: Optional[Path] = None) -> dict[str, Any]:
    """Load strategy configuration (toggle + params per strategy)."""
    path = path or CONFIG_DIR / "strategies.yaml"
    raw = _load_yaml(path)
    return raw.get("strategies", {})


def setup_logging(path: Optional[Path] = None) -> None:
    """Initialise logging from YAML config."""
    path = path or CONFIG_DIR / "logging.yaml"
    # Ensure log directory exists
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    config = _load_yaml(path)
    logging.config.dictConfig(config)
    logger.info("Logging configured from %s", path)
