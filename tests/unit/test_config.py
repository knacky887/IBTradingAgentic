"""
TradingApp — Unit Tests for Configuration Loader
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.core.config import (
    AppSettings,
    BrokerSettings,
    RedisSettings,
    UniverseSettings,
    RiskSettings,
    PortfolioSettings,
    ExecutionSettings,
    DataSettings,
    SystemSettings,
    load_settings,
    load_strategy_config,
    setup_logging,
    _load_yaml,
    _apply_env_overrides,
    PROJECT_ROOT,
    CONFIG_DIR,
)


class TestSettingsModels:
    def test_broker_defaults(self):
        s = BrokerSettings()
        assert s.name == "interactive_brokers"
        assert s.port == 7497

    def test_redis_defaults(self):
        s = RedisSettings()
        assert s.host == "127.0.0.1"
        assert s.port == 6379

    def test_universe_defaults(self):
        s = UniverseSettings()
        assert s.source == "sp500"
        assert s.custom_symbols == []

    def test_risk_defaults(self):
        s = RiskSettings()
        assert s.max_position_pct == 0.05
        assert s.var_confidence == 0.99

    def test_portfolio_defaults(self):
        s = PortfolioSettings()
        assert s.rebalance_cadence == "EOD"
        assert s.max_positions == 50

    def test_execution_defaults(self):
        s = ExecutionSettings()
        assert s.default_order_type == "LIMIT"
        assert s.max_slippage_bps == 10

    def test_data_defaults(self):
        s = DataSettings()
        assert s.bar_size == "1 min"
        assert s.history_days == 504

    def test_system_defaults(self):
        s = SystemSettings()
        assert s.mode == "paper"
        assert s.log_level == "INFO"

    def test_app_settings_all_defaults(self):
        s = AppSettings()
        assert s.broker.port == 7497
        assert s.redis.port == 6379
        assert s.risk.max_position_pct == 0.05

    def test_app_settings_custom(self):
        s = AppSettings(broker=BrokerSettings(port=4002))
        assert s.broker.port == 4002


class TestYamlLoader:
    def test_load_yaml_valid(self, tmp_path):
        p = tmp_path / "test.yaml"
        p.write_text(yaml.dump({"key": "value", "num": 42}))
        data = _load_yaml(p)
        assert data["key"] == "value"
        assert data["num"] == 42

    def test_load_yaml_empty(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        data = _load_yaml(p)
        assert data == {}


class TestEnvOverrides:
    def test_env_override_string(self):
        raw = {"broker": {"host": "localhost", "port": 7497}}
        os.environ["TRADINGAPP_BROKER_HOST"] = "192.168.1.1"
        try:
            result = _apply_env_overrides(raw)
            assert result["broker"]["host"] == "192.168.1.1"
        finally:
            del os.environ["TRADINGAPP_BROKER_HOST"]

    def test_env_override_int(self):
        raw = {"broker": {"host": "localhost", "port": 7497}}
        os.environ["TRADINGAPP_BROKER_PORT"] = "4002"
        try:
            result = _apply_env_overrides(raw)
            assert result["broker"]["port"] == 4002
        finally:
            del os.environ["TRADINGAPP_BROKER_PORT"]

    def test_env_override_float(self):
        raw = {"risk": {"max_position_pct": 0.05}}
        os.environ["TRADINGAPP_RISK_MAX_POSITION_PCT"] = "0.10"
        try:
            result = _apply_env_overrides(raw)
            assert result["risk"]["max_position_pct"] == 0.10
        finally:
            del os.environ["TRADINGAPP_RISK_MAX_POSITION_PCT"]

    def test_unrelated_env_var_ignored(self):
        raw = {"broker": {"port": 7497}}
        os.environ["OTHER_VAR"] = "ignored"
        try:
            result = _apply_env_overrides(raw)
            assert result["broker"]["port"] == 7497
        finally:
            del os.environ["OTHER_VAR"]

    def test_env_missing_section_ignored(self):
        raw = {"broker": {"port": 7497}}
        os.environ["TRADINGAPP_NONEXISTENT_KEY"] = "val"
        try:
            result = _apply_env_overrides(raw)
            assert "nonexistent" not in result
        finally:
            del os.environ["TRADINGAPP_NONEXISTENT_KEY"]


class TestLoadSettings:
    def test_load_from_project_config(self):
        cfg = CONFIG_DIR / "settings.yaml"
        if cfg.exists():
            settings = load_settings(cfg)
            assert isinstance(settings, AppSettings)

    def test_load_custom_yaml(self, tmp_path):
        custom = tmp_path / "settings.yaml"
        custom.write_text(yaml.dump({
            "system": {"mode": "live"},
            "broker": {"port": 4001},
        }))
        settings = load_settings(custom)
        assert settings.system.mode == "live"
        assert settings.broker.port == 4001


class TestLoadStrategyConfig:
    def test_load_strategy_config_from_project(self):
        cfg = CONFIG_DIR / "strategies.yaml"
        if cfg.exists():
            strats = load_strategy_config(cfg)
            assert isinstance(strats, dict)

    def test_load_empty_strategy_config(self, tmp_path):
        p = tmp_path / "strategies.yaml"
        p.write_text(yaml.dump({"strategies": {}}))
        strats = load_strategy_config(p)
        assert strats == {}

    def test_load_no_strategies_key(self, tmp_path):
        p = tmp_path / "strategies.yaml"
        p.write_text(yaml.dump({"other_key": True}))
        strats = load_strategy_config(p)
        assert strats == {}


class TestSetupLogging:
    def test_setup_logging(self):
        cfg = CONFIG_DIR / "logging.yaml"
        if cfg.exists():
            setup_logging(cfg)  # Should not raise


class TestProjectPaths:
    def test_config_dir_exists(self):
        assert CONFIG_DIR.exists()

    def test_project_root_is_parent_of_config(self):
        assert CONFIG_DIR.parent == PROJECT_ROOT
