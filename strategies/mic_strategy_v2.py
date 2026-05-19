import pandas as pd
import numpy as np

class MultiIndicatorConfluenceStrategyV2:
    name = "MultiIndicatorConfluenceV2"
    description = "Optimized MIC with stricter trend and volume filters"

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.atr_sl_mult   = self.config.get('atr_sl_mult', 2.0)
        self.atr_tp_mult   = self.config.get('atr_tp_mult', 4.0)
        self.rsi_low       = self.config.get('rsi_low', 45)
        self.rsi_high      = self.config.get('rsi_high', 60)
        self.vol_min       = self.config.get('vol_min', 1.5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Stricter trend: EMA8 > EMA34 AND Price > EMA200 AND EMA200 is rising
        trend_bull = (df['ema8'] > df['ema34']) & (df['ema34'] > df['ema55'])
        macro_bull = (df['close'] > df['ema200']) & (df['ema200'] > df['ema200'].shift(5))
        
        # Momentum: MACD hist rising and positive
        macd_bull  = (df['macd_hist'] > 0) & (df['macd_hist'] > df['macd_hist'].shift(1))
        
        # RSI: Not overbought, but showing strength
        rsi_ok     = (df['rsi'] >= self.rsi_low) & (df['rsi'] <= self.rsi_high)
        
        # Volume: Significant spike
        vol_ok     = df['volume_ratio'] >= self.vol_min

        # ADX: Strong trend
        adx_ok = df['adx'] > 25

        buy_signal = trend_bull & macro_bull & macd_bull & rsi_ok & vol_ok & adx_ok

        # Exit: Trend reversal or overbought
        sell_signal = (df['ema8'] < df['ema34']) | (df['rsi'] > 75) | (df['macd_hist'] < 0)

        df['signal'] = 0
        df.loc[buy_signal, 'signal']  = 1
        df.loc[sell_signal, 'signal'] = -1

        df['sl'] = df['close'] - self.atr_sl_mult * df['atr']
        df['tp'] = df['close'] + self.atr_tp_mult * df['atr']

        return df
