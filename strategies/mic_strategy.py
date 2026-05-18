"""
Strategy 1: Multi-Indicator Confluence (MIC)
============================================
Based on:
- IEEE paper: EMA strategy with profit factor 3.5, win rate 60%, risk-reward 2.2
- Multi-indicator confluence: enter only when ALL conditions agree

Entry (Long):
  - EMA8 > EMA34 (bullish trend)
  - MACD histogram > 0 and rising (momentum confirms)
  - RSI 45-65 (not overbought, but positive)
  - Price above EMA200 (macro uptrend)
  - Volume ratio > 1.0 (volume confirms)

Exit:
  - Stop-loss: 1.5× ATR below entry
  - Take-profit: 3× ATR above entry (2:1 R:R)
  - EMA crossdown (trend reversal)
  - RSI > 70 (overbought exit)
"""
import pandas as pd
import numpy as np


class MultiIndicatorConfluenceStrategy:
    name = "MultiIndicatorConfluence"
    description = "EMA+RSI+MACD+ATR confluence — IEEE paper proven (profit factor 3.5)"

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.atr_sl_mult   = self.config.get('atr_sl_mult', 1.5)
        self.atr_tp_mult   = self.config.get('atr_tp_mult', 3.0)
        self.rsi_low       = self.config.get('rsi_low', 45)
        self.rsi_high      = self.config.get('rsi_high', 65)
        self.vol_min       = self.config.get('vol_min', 1.0)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate BUY/SELL/HOLD signals. Returns df with 'signal', 'sl', 'tp' columns."""
        df = df.copy()

        # ── Entry conditions ──────────────────────────────────────────
        trend_bull = df['ema8_34_cross'] == 1
        macro_bull = df['price_above_200'] == 1
        macd_bull  = (df['macd_hist'] > 0) & (df['macd_hist_rising'] == 1)
        rsi_ok     = (df['rsi'] >= self.rsi_low) & (df['rsi'] <= self.rsi_high)
        vol_ok     = df['volume_ratio'] >= self.vol_min

        # All conditions must agree
        buy_signal = trend_bull & macro_bull & macd_bull & rsi_ok & vol_ok

        # Require EMA cross actually happened recently (within last 3 candles)
        ema_just_crossed = (
            (df['ema8_34_cross'] == 1) &
            (df['ema8_34_cross'].shift(1) == 0)
        ) | (
            (df['ema8_34_cross'] == 1) &
            (df['ema8_34_cross'].shift(2) == 0)
        )
        # Also allow continuation signals when trend is strong
        strong_trend_buy = trend_bull & macro_bull & macd_bull & rsi_ok & (df['adx'] > 30)

        buy_signal = ema_just_crossed & buy_signal | strong_trend_buy & vol_ok

        # ── Exit conditions ───────────────────────────────────────────
        sell_signal = (
            (df['ema8_34_cross'] == 0) |  # trend reversal
            (df['rsi'] > 70) |            # overbought
            (df['macd_hist'] < 0)         # momentum flipped
        )

        # ── Build signal column ───────────────────────────────────────
        df['signal'] = 0
        df.loc[buy_signal, 'signal']  = 1
        df.loc[sell_signal, 'signal'] = -1

        # ── Stop-loss and Take-profit ─────────────────────────────────
        df['sl'] = df['close'] - self.atr_sl_mult * df['atr']
        df['tp'] = df['close'] + self.atr_tp_mult * df['atr']

        return df
