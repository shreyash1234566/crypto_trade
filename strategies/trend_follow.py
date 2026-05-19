import pandas as pd
import numpy as np

class TrendFollowStrategy:
    name = "TrendFollow"
    description = "Simple EMA cross with ADX filter"

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.atr_sl_mult = 2.0
        self.atr_tp_mult = 6.0 # High R:R

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # EMA Cross
        buy_signal = (df['ema8'] > df['ema21']) & (df['ema8'].shift(1) <= df['ema21'].shift(1))
        # Filter: ADX > 25 and Price > EMA200
        buy_signal &= (df['adx'] > 25) & (df['close'] > df['ema200'])
        
        sell_signal = (df['ema8'] < df['ema21'])
        
        df['signal'] = 0
        df.loc[buy_signal, 'signal']  = 1
        df.loc[sell_signal, 'signal'] = -1
        
        df['sl'] = df['close'] - self.atr_sl_mult * df['atr']
        df['tp'] = df['close'] + self.atr_tp_mult * df['atr']
        
        return df
