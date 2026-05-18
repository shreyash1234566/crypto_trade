"""
Strategy 2: MACD-ADX Trend Momentum
=====================================
Based on research paper arXiv:2511.00665:
- MACD optimal params: Short=17, Long=21, Signal=15
- ADX=13 for trend strength confirmation
- Achieved best results in BTC backtesting 2024-01-10 to 2024-12-31
- Focuses on quality over quantity (fewer, better trades)

Entry (Long):
  - MACD optimized > signal (macd_opt > macd_opt_signal)
  - ADX > 20 (trend exists, uses research optimal=13 but 20 for safety)
  - DI+ > DI- (directional indicator bullish)
  - Price above EMA55 (intermediate trend)
  - RSI not overbought (< 70)

Short/Exit:
  - MACD cross below signal
  - ADX weakening + DI- > DI+
  - RSI divergence
"""
import pandas as pd
import numpy as np


class MACDADXStrategy:
    name = "MACD_ADX_Trend"
    description = "Research-optimal MACD(17,21,15)+ADX trend momentum strategy"

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.adx_threshold = self.config.get('adx_threshold', 20)
        self.atr_sl_mult   = self.config.get('atr_sl_mult', 2.0)
        self.atr_tp_mult   = self.config.get('atr_tp_mult', 3.5)
        self.rsi_cap       = self.config.get('rsi_cap', 70)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── Use research-optimal MACD (17, 21, 15) ───────────────────
        macd_bull_opt = df['macd_opt'] > df['macd_opt_signal']

        # ── ADX trend strength ────────────────────────────────────────
        trend_strong = df['adx'] >= self.adx_threshold
        di_bull = df['di_plus'] > df['di_minus']

        # ── Price above medium-term EMA ───────────────────────────────
        price_ok = df['close'] > df['ema55']

        # ── RSI not extreme ───────────────────────────────────────────
        rsi_ok = df['rsi'] < self.rsi_cap

        # ── MACD just crossed up ──────────────────────────────────────
        macd_just_crossed = (
            (df['macd_opt'] > df['macd_opt_signal']) &
            (df['macd_opt'].shift(1) <= df['macd_opt_signal'].shift(1))
        )

        # ── Trend continuation (ADX rising, DI+ > DI-) ───────────────
        adx_rising = df['adx'] > df['adx'].shift(3)
        trend_cont = macd_bull_opt & trend_strong & di_bull & adx_rising

        # ── Combined entry ────────────────────────────────────────────
        buy_signal = (macd_just_crossed & trend_strong & di_bull & price_ok & rsi_ok) | \
                     (trend_cont & price_ok & rsi_ok & (df['volume_ratio'] > 1.2))

        # ── Exit ──────────────────────────────────────────────────────
        macd_cross_down = (
            (df['macd_opt'] < df['macd_opt_signal']) &
            (df['macd_opt'].shift(1) >= df['macd_opt_signal'].shift(1))
        )
        sell_signal = macd_cross_down | (df['di_minus'] > df['di_plus']) | (df['rsi'] > 75)

        # ── Signals ───────────────────────────────────────────────────
        df['signal'] = 0
        df.loc[buy_signal, 'signal']  = 1
        df.loc[sell_signal, 'signal'] = -1

        df['sl'] = df['close'] - self.atr_sl_mult * df['atr']
        df['tp'] = df['close'] + self.atr_tp_mult * df['atr']

        return df
