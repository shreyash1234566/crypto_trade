import pandas as pd
import numpy as np
from data.fetcher import fetch_ohlcv
from data.features import add_all_features
from strategies.bb_rsi_strategy import BBRSIMeanReversionStrategy
from backtest.engine import BacktestEngine, BacktestConfig

def optimize():
    df = fetch_ohlcv('BTC/USDT', '1h', days=365)
    df = add_all_features(df)
    
    # Test a few promising variations
    configs = [
        {'rsi_oversold': 30, 'stoch_oversold': 20, 'bb_pct_entry': 0.1, 'atr_sl_mult': 1.0},
        {'rsi_oversold': 40, 'stoch_oversold': 30, 'bb_pct_entry': 0.2, 'atr_sl_mult': 2.0},
        {'rsi_oversold': 35, 'stoch_oversold': 25, 'bb_pct_entry': 0.15, 'atr_sl_mult': 1.5},
        {'rsi_oversold': 45, 'stoch_oversold': 35, 'bb_pct_entry': 0.25, 'atr_sl_mult': 2.5},
    ]
    
    for config in configs:
        strategy = BBRSIMeanReversionStrategy(config)
        signals = strategy.generate_signals(df)
        bt_config = BacktestConfig(initial_capital=10000, risk_per_trade=0.02)
        engine = BacktestEngine(bt_config)
        result = engine.run(df, signals)
        ret = result['metrics'].get('total_return_pct', -100)
        print(f"Return: {ret}% with {config}")

if __name__ == '__main__':
    optimize()
