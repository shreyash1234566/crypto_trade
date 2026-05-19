import pandas as pd
import numpy as np
from data.fetcher import fetch_ohlcv
from data.features import add_all_features
from strategies.bb_rsi_strategy import BBRSIMeanReversionStrategy
from backtest.engine import BacktestEngine, BacktestConfig
import itertools

def optimize():
    # Fetch data
    df = fetch_ohlcv('BTC/USDT', '1h', days=365)
    df = add_all_features(df)
    
    # Parameter grid
    rsi_oversolds = [30, 35, 40]
    stoch_oversolds = [20, 25, 30]
    bb_pct_entries = [0.1, 0.15, 0.2]
    atr_sl_mults = [1.0, 1.5, 2.0]
    
    best_return = -np.inf
    best_params = None
    
    for rsi, stoch, bb, sl in itertools.product(rsi_oversolds, stoch_oversolds, bb_pct_entries, atr_sl_mults):
        config = {
            'rsi_oversold': rsi,
            'stoch_oversold': stoch,
            'bb_pct_entry': bb,
            'atr_sl_mult': sl
        }
        
        strategy = BBRSIMeanReversionStrategy(config)
        signals = strategy.generate_signals(df)
        
        bt_config = BacktestConfig(initial_capital=10000, risk_per_trade=0.02)
        engine = BacktestEngine(bt_config)
        result = engine.run(df, signals)
        
        ret = result['metrics'].get('total_return_pct', -100)
        
        if ret > best_return:
            best_return = ret
            best_params = config
            print(f"New best: {ret}% with {config}")

    print("\nOptimization Finished")
    print(f"Best Return: {best_return}%")
    print(f"Best Params: {best_params}")

if __name__ == '__main__':
    optimize()
