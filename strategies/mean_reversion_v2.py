import pandas as pd
import numpy as np

class MeanReversionStrategyV2:
    name = "MeanReversionV2"
    description = "Buy extreme oversold dips"

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.atr_sl_mult = 2.0
        self.atr_tp_mult = 2.0

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Extreme oversold: RSI < 25 and Price < Lower BB
        buy_signal = (df['rsi'] < 25) & (df['close'] < df['bb_lower'])
        
        # Exit when RSI recovers to 50 or hits Mid BB
        sell_signal = (df['rsi'] > 50) | (df['close'] > df['bb_mid'])
        
        df['signal'] = 0
        df.loc[buy_signal, 'signal']  = 1
        df.loc[sell_signal, 'signal'] = -1
        
        df['sl'] = df['close'] - self.atr_sl_mult * df['atr']
        df['tp'] = df['close'] + self.atr_tp_mult * df['atr']
        
        return df
