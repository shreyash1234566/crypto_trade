import pandas as pd
import numpy as np
from data.fetcher import fetch_ohlcv
from data.features import add_all_features
from strategies.optimized_trend import OptimizedTrendStrategy
from backtest.engine import BacktestEngine, BacktestConfig
from main import _plot_equity

def run():
    df = fetch_ohlcv('BTC/USDT', '1h', days=365)
    df = add_all_features(df)
    
    strategy = OptimizedTrendStrategy()
    signals = strategy.generate_signals(df)
    
    bt_config = BacktestConfig(initial_capital=10000, risk_per_trade=0.1)
    engine = BacktestEngine(bt_config)
    result = engine.run(df, signals)
    
    _plot_equity(result['equity_curve'], df, strategy.name, 'BTC/USDT')

if __name__ == '__main__':
    run()
