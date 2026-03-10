"""
TradingApp — Backtest CLI Runner

Usage:
  python scripts/run_backtest.py --data data/parquet/SPY.parquet --strategy dual_ma_crossover
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_strategy_config, setup_logging
from src.utils.data_store import load_parquet
from backtest.engine import BacktestEngine


def main():
    parser = argparse.ArgumentParser(description="TradingApp Backtest Runner")
    parser.add_argument("--data", required=True, help="Path to Parquet data file")
    parser.add_argument("--strategy", help="Run only this strategy (name from strategies.yaml)")
    parser.add_argument("--nav", type=float, default=1_000_000, help="Initial NAV")
    parser.add_argument("--output", help="Output JSON file for results")
    args = parser.parse_args()

    # Setup
    setup_logging()
    logger = logging.getLogger("tradingapp.backtest")

    # Load strategy config
    strategy_config = load_strategy_config()

    # If a specific strategy is requested, disable all others
    if args.strategy:
        for name, cfg in strategy_config.items():
            cfg["enabled"] = (name == args.strategy)

    # Load data
    logger.info("Loading data from %s", args.data)
    bars = load_parquet(args.data)
    logger.info("Loaded %d bars", len(bars))

    # Run backtest
    engine = BacktestEngine(strategy_config=strategy_config, initial_nav=args.nav)
    equity_df = asyncio.run(engine.run(bars))

    # Stats
    stats = engine.compute_stats(equity_df)
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:>12.4f}")
        else:
            print(f"  {key:30s}: {value:>12}")
    print("=" * 60)

    # Save
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
