import pandas as pd
import numpy as np

class OptimizedTrendStrategy:
    name = "OptimizedTrend"
    description = "EMA cross with macro filter and high R:R"

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.atr_sl_mult = 1.5
        self.atr_tp_mult = 8.0

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Entry: EMA8 crosses above EMA21 in a macro uptrend (Price > EMA200)
        buy_signal = (df['ema8'] > df['ema21']) & (df['ema8'].shift(1) <= df['ema21'].shift(1))
        buy_signal &= (df['close'] > df['ema200'])
        
        # Exit: Price drops below EMA21 (trailing stop proxy)
        sell_signal = (df['close'] < df['ema21'])
        
        df['signal'] = 0
        df.loc[buy_signal, 'signal']  = 1
        df.loc[sell_signal, 'signal'] = -1
        
        df['sl'] = df['close'] - self.atr_sl_mult * df['atr']
        df['tp'] = df['close'] + self.atr_tp_mult * df['atr']
        
        return df
