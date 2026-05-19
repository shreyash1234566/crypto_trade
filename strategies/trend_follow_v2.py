import pandas as pd
import numpy as np

class TrendFollowStrategyV2:
    name = "TrendFollowV2"
    description = "EMA cross with ADX and Volume filter, high R:R"

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.atr_sl_mult = 1.5
        self.atr_tp_mult = 5.0 

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # EMA Cross: 8 over 21
        buy_signal = (df['ema8'] > df['ema21']) & (df['ema8'].shift(1) <= df['ema21'].shift(1))
        
        # Looser Filters
        buy_signal &= (df['adx'] > 20) & (df['volume_ratio'] > 1.1)
        
        # Exit on EMA cross back
        sell_signal = (df['ema8'] < df['ema21'])
        
        df['signal'] = 0
        df.loc[buy_signal, 'signal']  = 1
        df.loc[sell_signal, 'signal'] = -1
        
        df['sl'] = df['close'] - self.atr_sl_mult * df['atr']
        df['tp'] = df['close'] + self.atr_tp_mult * df['atr']
        
        return df
