import pandas as pd
from data.fetcher import fetch_ohlcv
from data.features import add_all_features
from strategies.optimized_trend import OptimizedTrendStrategy
from backtest.engine import BacktestEngine, BacktestConfig

def test():
    df = fetch_ohlcv('BTC/USDT', '1h', days=365)
    df = add_all_features(df)
    
    strategy = OptimizedTrendStrategy()
    signals = strategy.generate_signals(df)
    
    bt_config = BacktestConfig(initial_capital=10000, risk_per_trade=0.1)
    engine = BacktestEngine(bt_config)
    result = engine.run(df, signals)
    
    print(f"Final Capital: {result['final_capital']}")
    if 'error' not in result['metrics']:
        print(f"Total Return: {result['metrics']['total_return_pct']}%")
        print(f"Win Rate: {result['metrics']['win_rate']}%")
        print(f"Trades: {result['metrics']['total_trades']}")
    else:
        print("No trades executed.")

if __name__ == '__main__':
    test()
