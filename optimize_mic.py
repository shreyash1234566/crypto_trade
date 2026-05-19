import pandas as pd
import numpy as np
from data.fetcher import fetch_ohlcv
from data.features import add_all_features
from strategies.mic_strategy import MultiIndicatorConfluenceStrategy
from backtest.engine import BacktestEngine, BacktestConfig
import itertools

def optimize():
    # Fetch data
    df = fetch_ohlcv('BTC/USDT', '1h', days=365)
    df = add_all_features(df)
    
    # Parameter grid
    atr_sl_mults = [1.0, 1.5, 2.0]
    atr_tp_mults = [2.0, 3.0, 4.0]
    rsi_lows = [40, 45, 50]
    rsi_highs = [60, 65, 70]
    
    best_return = -np.inf
    best_params = None
    
    results = []
    
    for sl, tp, r_low, r_high in itertools.product(atr_sl_mults, atr_tp_mults, rsi_lows, rsi_highs):
        config = {
            'atr_sl_mult': sl,
            'atr_tp_mult': tp,
            'rsi_low': r_low,
            'rsi_high': r_high
        }
        
        strategy = MultiIndicatorConfluenceStrategy(config)
        signals = strategy.generate_signals(df)
        
        bt_config = BacktestConfig(initial_capital=10000, risk_per_trade=0.02)
        engine = BacktestEngine(bt_config)
        result = engine.run(df, signals)
        
        ret = result['metrics'].get('total_return_pct', -100)
        results.append((config, ret))
        
        if ret > best_return:
            best_return = ret
            best_params = config
            print(f"New best: {ret}% with {config}")

    print("\nOptimization Finished")
    print(f"Best Return: {best_return}%")
    print(f"Best Params: {best_params}")

if __name__ == '__main__':
    optimize()
