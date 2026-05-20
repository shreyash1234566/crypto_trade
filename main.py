#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║          CRYPTO TRADING PRO — Research-Backed Trading System         ║
║                                                                      ║
║  Strategies:                                                         ║
║   1. Multi-Indicator Confluence (EMA+RSI+MACD+ATR)                  ║
║      → IEEE paper: profit factor 3.5, win rate 60%                  ║
║   2. MACD-ADX Trend Momentum                                         ║
║      → arXiv:2511.00665: optimal params for BTC/USDT 2024           ║
║   3. BB-RSI Mean Reversion                                           ║
║      → ClucMay72018: 80% win rate, Sharpe 1.84                      ║
║   4. ML-Enhanced (LogReg rolling window filter)                      ║
║      → Research paper: reduces false positives by 30-40%            ║
║                                                                      ║
║  Features:                                                           ║
║   - No lookahead bias (signals on close, execute on next open)       ║
║   - Realistic fees (0.1%) + slippage (0.05%)                        ║
║   - Risk-based position sizing (2% per trade)                        ║
║   - Walk-forward validation (anti-overfitting)                       ║
║   - Comprehensive performance reporting                              ║
╚══════════════════════════════════════════════════════════════════════╝

Usage:
  python main.py --symbol BTC/USDT --timeframe 1h --days 365 --mode backtest
  python main.py --mode compare      # compare all strategies
  python main.py --mode paper        # paper trading on latest data
"""
import argparse
import sys
import os
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         🚀 CRYPTO TRADING PRO — Research-Backed System          ║
╚══════════════════════════════════════════════════════════════════╝
""")


def run_backtest(symbol: str, timeframe: str, days: int,
                 strategy_name: str, use_ml: bool,
                 walk_forward: bool, config: dict,
                 precision_floor: float = 0.80, weight_scale: float = 1.0) -> dict:
    """Run a full backtest and return results."""
    from data.fetcher import fetch_ohlcv
    from data.features import add_all_features
    from backtest.engine import BacktestEngine, BacktestConfig, walk_forward_backtest

    # ── Load data ──────────────────────────────────────────────────────
    print(f"\n[1/4] Fetching {symbol} {timeframe} data ({days} days)...")
    df = fetch_ohlcv(symbol, timeframe, days)
    print(f"      → {len(df):,} candles from {df.index[0].date()} to {df.index[-1].date()}")

    # ── Add features ───────────────────────────────────────────────────
    print(f"[2/4] Computing technical indicators...")
    df = add_all_features(df)
    print(f"      → {len(df):,} candles after indicator warmup")

    # ── Load strategy ──────────────────────────────────────────────────
    print(f"[3/4] Generating signals with {strategy_name}...")
    strategy = _load_strategy(strategy_name, config)

    if use_ml:
        from models.ml_enhancer import MLSignalEnhancer
        ml_config = {'precision_floor': precision_floor, 'weight_scale': weight_scale}
        strategy = MLSignalEnhancer(strategy, config=ml_config)
        print(f"      → ML enhancement enabled (LogisticRegression rolling filter, precision floor={precision_floor}, weight scale={weight_scale})")

    # ── Backtest ───────────────────────────────────────────────────────
    bt_config = BacktestConfig(
        initial_capital=config.get('capital', 10000),
        fee_rate=config.get('fee_rate', 0.001),
        slippage_rate=config.get('slippage', 0.0005),
        risk_per_trade=config.get('risk', 0.02),
        max_open_trades=1,
    )

    print(f"[4/4] Running {'walk-forward' if walk_forward else 'full'} backtest...")

    if walk_forward:
        result = walk_forward_backtest(df, strategy, bt_config,
                                       train_size=config.get('train_days', 30),
                                       test_size=config.get('test_days', 7),
                                       step_size=config.get('step_days', 7))
        result['df'] = df
        result['strategy_name'] = strategy.name
    else:
        signals = strategy.generate_signals(df)
        engine = BacktestEngine(bt_config)
        result = engine.run(df, signals)
        result['df'] = df
        result['strategy_name'] = strategy.name

    return result


def _load_strategy(name: str, config: dict = None):
    """Load a strategy by name."""
    if name == 'mic':
        from strategies.mic_strategy import MultiIndicatorConfluenceStrategy
        return MultiIndicatorConfluenceStrategy(config)
    elif name == 'macd_adx':
        from strategies.macd_adx_strategy import MACDADXStrategy
        return MACDADXStrategy(config)
    elif name == 'bb_rsi':
        from strategies.bb_rsi_strategy import BBRSIMeanReversionStrategy
        return BBRSIMeanReversionStrategy(config)
    else:
        raise ValueError(f"Unknown strategy: {name}. Use: mic, macd_adx, bb_rsi")


def cmd_backtest(args):
    """Run single strategy backtest."""
    result = run_backtest(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        strategy_name=args.strategy,
        use_ml=args.ml,
        walk_forward=args.walk_forward,
        config={'capital': args.capital},
        precision_floor=args.precision_floor,
        weight_scale=args.weight_scale,
    )

    from reports.reporter import generate_report
    df = result.pop('df')
    name = result.pop('strategy_name')

    report = generate_report(result, name, args.symbol, args.timeframe, df,
                              output_dir=Path('reports'))
    print("\n" + report)

    # Save equity curve
    if 'equity_curve' in result:
        _plot_equity(result['equity_curve'], df, name, args.symbol)


def cmd_compare(args):
    """Compare all strategies."""
    all_results = {}
    strategy_names = ['mic', 'macd_adx', 'bb_rsi']
    use_ml = [False, False, True]  # Apply ML to bb_rsi

    for i, strat in enumerate(strategy_names):
        print(f"\n{'─'*60}")
        print(f"  Running strategy: {strat.upper()}")
        print(f"{'─'*60}")
        try:
            result = run_backtest(
                symbol=args.symbol,
                timeframe=args.timeframe,
                days=args.days,
                strategy_name=strat,
                use_ml=use_ml[i],
                walk_forward=True,
                config={'capital': args.capital}
            )
            name = result.pop('strategy_name', strat)
            result.pop('df', None)
            all_results[name] = result
        except Exception as e:
            print(f"  [ERROR] {strat}: {e}")
            all_results[strat] = {'metrics': {'error': str(e)}}

    from reports.reporter import compare_strategies
    print("\n" + compare_strategies(all_results))

    # Find winner
    best_name, best_pf = None, -999
    for name, res in all_results.items():
        pf = res.get('avg_profit_factor', res.get('metrics', {}).get('profit_factor', 0))
        if pf > best_pf:
            best_pf, best_name = pf, name

    if best_name:
        print(f"\n  🏆 Best strategy: {best_name} (Profit Factor: {best_pf:.3f})")


def cmd_paper(args):
    """Paper trading on most recent data."""
    print("\n📋 PAPER TRADING MODE")
    print("   Running on latest candles. No real trades executed.\n")

    from data.fetcher import fetch_ohlcv
    from data.features import add_all_features

    # Use last 30 days for paper trading simulation
    df = fetch_ohlcv(args.symbol, args.timeframe, days=60, use_cache=False)
    df = add_all_features(df)

    # Use the last 30 days as "live" paper trading window
    paper_df = df.iloc[-30 * 24:]

    strategy = _load_strategy(args.strategy, {})
    signals = strategy.generate_signals(paper_df)

    from backtest.engine import BacktestEngine, BacktestConfig
    bt_config = BacktestConfig(
        initial_capital=args.capital,
        risk_per_trade=0.02,
    )
    engine = BacktestEngine(bt_config)
    result = engine.run(paper_df, signals)

    from reports.reporter import generate_report
    report = generate_report(result, strategy.name, args.symbol, args.timeframe,
                              paper_df, output_dir=Path('reports'))
    print(report)

    # Show recent signals
    recent = signals.iloc[-10:]
    buys = recent[recent['signal'] == 1]
    sells = recent[recent['signal'] == -1]

    print(f"\n  Recent BUY signals:  {len(buys)} in last 10 candles")
    print(f"  Recent SELL signals: {len(sells)} in last 10 candles")

    if not buys.empty:
        last_buy = buys.iloc[-1]
        print(f"\n  Last BUY signal:")
        print(f"    Time  : {buys.index[-1]}")
        print(f"    Price : ${paper_df.loc[buys.index[-1], 'close']:,.2f}")
        print(f"    SL    : ${last_buy['sl']:,.2f}")
        print(f"    TP    : ${last_buy['tp']:,.2f}")


def _plot_equity(equity_curve: list, df: pd.DataFrame, name: str, symbol: str):
    """Save a simple equity curve chart."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})
        fig.patch.set_facecolor('#0d1117')

        # Equity curve
        eq = np.array(equity_curve)
        # Align with df
        n = min(len(eq), len(df))
        dates = df.index[:n]

        ax1.set_facecolor('#0d1117')
        ax1.plot(dates, eq[:n], color='#00ff88', linewidth=1.5, label='Portfolio Value')
        ax1.axhline(y=eq[0], color='#888', linestyle='--', linewidth=0.8, alpha=0.7, label='Initial Capital')
        ax1.fill_between(dates, eq[0], eq[:n], where=eq[:n] > eq[0],
                         alpha=0.2, color='#00ff88', label='Profit Zone')
        ax1.fill_between(dates, eq[0], eq[:n], where=eq[:n] < eq[0],
                         alpha=0.2, color='#ff4444', label='Loss Zone')
        ax1.set_title(f'{name} — {symbol} Equity Curve', color='white', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value (USD)', color='#aaa')
        ax1.tick_params(colors='#aaa')
        ax1.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='white', fontsize=9)
        ax1.grid(True, alpha=0.15, color='#444')
        for spine in ax1.spines.values():
            spine.set_color('#333')

        # Price
        ax2.set_facecolor('#0d1117')
        ax2.plot(df.index, df['close'], color='#4488ff', linewidth=0.8)
        ax2.set_title(f'{symbol} Price', color='white', fontsize=11)
        ax2.set_ylabel('Price (USD)', color='#aaa')
        ax2.tick_params(colors='#aaa')
        ax2.grid(True, alpha=0.15, color='#444')
        for spine in ax2.spines.values():
            spine.set_color('#333')

        plt.tight_layout()
        out = Path('reports') / f"equity_{name}_{symbol.replace('/', '_')}.png"
        out.parent.mkdir(exist_ok=True)
        plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='#0d1117')
        plt.close()
        print(f"\n[Chart] Equity curve saved to {out}")
    except Exception as e:
        print(f"[Chart] Could not save chart: {e}")


def main():
    banner()
    parser = argparse.ArgumentParser(
        description='Crypto Trading Pro — Research-Backed System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--symbol',    default='BTC/USDT', help='Trading pair (default: BTC/USDT)')
    parser.add_argument('--timeframe', default='1h',       help='Timeframe (default: 1h)')
    parser.add_argument('--days',      type=int, default=365, help='Days of data (default: 365)')
    parser.add_argument('--capital',   type=float, default=10000, help='Initial capital USD (default: 10000)')
    parser.add_argument('--strategy',  default='mic',
                        choices=['mic', 'macd_adx', 'bb_rsi'],
                        help='Strategy (default: mic)')
    parser.add_argument('--mode',      default='backtest',
                        choices=['backtest', 'compare', 'paper'],
                        help='Run mode (default: backtest)')
    parser.add_argument('--ml',        action='store_true', help='Enable ML signal filter')
    parser.add_argument('--precision-floor', type=float, default=0.80, help='Target minimum precision for ML filter')
    parser.add_argument('--weight-scale', type=float, default=1.0, help='Scale for positive class weight in ML filter')
    parser.add_argument('--walk-forward', action='store_true', help='Use walk-forward validation')

    args = parser.parse_args()

    Path('reports').mkdir(exist_ok=True)

    if args.mode == 'backtest':
        cmd_backtest(args)
    elif args.mode == 'compare':
        cmd_compare(args)
    elif args.mode == 'paper':
        cmd_paper(args)


if __name__ == "__main__":
    main()
