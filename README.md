# TradingApp — Multi-Agent Algorithmic Trading System

An institutional-grade, fully autonomous trading system built on a **six-agent architecture** communicating over **Redis Pub/Sub**. Targets US Equities via Interactive Brokers.

## Architecture

```
Market Data → [Data Ingestion] → Redis Bus → [Alpha Generation] → [Portfolio Mgmt]
                                                                        ↓
                  [System Monitor] ← Redis Bus ← [Execution] ← [Risk Management]
```

| Agent | Role |
|-------|------|
| **Data Ingestion** | Normalise IB bars, publish to bus, persist to Parquet |
| **Alpha Generation** | Run 25 quantitative strategies, emit signals |
| **Portfolio Management** | Aggregate signals, compute target portfolio, propose orders |
| **Risk Management** | Pre-trade gate (position limits, drawdown, exposure) |
| **Execution** | Smart order routing to IB (Limit, TWAP, VWAP) |
| **System Monitor** | Heartbeat checks, metrics, alerting |

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv && venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start infrastructure (Redis, Grafana, Prometheus)
docker-compose up -d

# 4. Run unit tests
python -m pytest tests/unit/ -v

# 5. Run a backtest
python scripts/run_backtest.py --data data/parquet/SPY.parquet --strategy dual_ma_crossover
```

## Strategy Families

- **Statistical Arbitrage** — Engle-Granger, Johansen, Kalman Filter, PCA, Copula
- **Mean Reversion** — Bollinger Bands, RSI-2, Ornstein-Uhlenbeck, VWAP, Hurst Filter
- **Trend Following** — Dual MA, Donchian, TSMOM, Kaufman AMA, LinReg Channel
- **Cross-Sectional** — Momentum Decile, Short-Term Reversal, Piotroski, Low-Vol, SUE
- **Volatility** — HMM Regime, Dispersion, Mean-Variance Overlay

## Configuration

All settings in `config/`:
- `settings.yaml` — Broker, Redis, risk limits, execution params
- `strategies.yaml` — Enable/disable strategies, tune parameters
- `logging.yaml` — Structured JSON logging

## Project Structure

```
src/
├── core/           # Config, models, message bus, base agent
├── agents/         # Six agent implementations
├── strategies/     # Strategy families (trend, mean-rev, stat-arb, etc.)
└── utils/          # Data store, IB client, metrics helpers
backtest/           # Event-driven backtest engine
tests/              # Unit + integration tests
scripts/            # CLI runners
config/             # YAML configuration
```
