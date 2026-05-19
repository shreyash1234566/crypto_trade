import pandas as pd
import numpy as np

class MacroBreakoutStrategy:
    name = "MacroBreakout"
    description = "Buy breakouts in macro uptrend"

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.atr_sl_mult = 1.5
        self.atr_tp_mult = 6.0

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Macro uptrend
        macro_bull = df['close'] > df['ema200']
        
        # Breakout: Close > Upper BB and Volume spike
        breakout = (df['close'] > df['bb_upper']) & (df['volume_ratio'] > 1.5)
        
        # Momentum: RSI > 60
        momentum = df['rsi'] > 60
        
        buy_signal = macro_bull & breakout & momentum
        
        # Exit when price drops below EMA21 or RSI drops below 40
        sell_signal = (df['close'] < df['ema21']) | (df['rsi'] < 40)
        
        df['signal'] = 0
        df.loc[buy_signal, 'signal']  = 1
        df.loc[sell_signal, 'signal'] = -1
        
        df['sl'] = df['close'] - self.atr_sl_mult * df['atr']
        df['tp'] = df['close'] + self.atr_tp_mult * df['atr']
        
        return df
