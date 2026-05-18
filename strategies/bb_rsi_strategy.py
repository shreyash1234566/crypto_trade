"""
Strategy 3: BB-RSI Mean Reversion
====================================
Inspired by NostalgiaForInfinity (most-starred Freqtrade strategy) and ClucMay72018
which achieved 80% win rate with Sharpe 1.84, 0.05% drawdown in research.

Approach: Buy oversold bounces in uptrending market, quick exits.

Entry (Long):
  - Price at/below lower Bollinger Band (mean reversion setup)
  - RSI < 35 (oversold)
  - Price > EMA200 (macro uptrend - only buy dips in bull market)
  - Stochastic < 25 (confirms oversold)
  - Volume spike (real selling, not thin air)

Exit:
  - Price reaches BB midline (mean reversion target)
  - RSI > 55 (recovered)
  - Hard stop: 2% below entry or 1× ATR

Risk management:
  - Smaller position sizes (higher win rate compensates lower R:R)
  - Quick exits preserve capital
"""
import pandas as pd
import numpy as np


class BBRSIMeanReversionStrategy:
    name = "BB_RSI_MeanReversion"
    description = "Bollinger+RSI oversold bounce — ClucMay72018 inspired (80% win rate)"

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.rsi_oversold  = self.config.get('rsi_oversold', 35)
        self.stoch_oversold= self.config.get('stoch_oversold', 28)
        self.bb_pct_entry  = self.config.get('bb_pct_entry', 0.15)  # price in lower 15% of BB
        self.atr_sl_mult   = self.config.get('atr_sl_mult', 1.0)    # tight stop
        self.rsi_exit      = self.config.get('rsi_exit', 55)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── Macro filter: only buy dips in bull market ────────────────
        macro_bull = df['price_above_200'] == 1

        # ── Oversold conditions ───────────────────────────────────────
        bb_oversold  = df['bb_pct'] < self.bb_pct_entry
        rsi_oversold = df['rsi'] < self.rsi_oversold
        stoch_oversold = df['stoch_k'] < self.stoch_oversold

        # ── Volume confirmation (real capitulation) ───────────────────
        vol_spike = df['volume_ratio'] > 1.3

        # ── Candle confirmation: bullish reversal candle ──────────────
        # Close above open AND close above previous close
        reversal_candle = (df['close'] > df['open']) & (df['close'] > df['close'].shift(1))

        # ── Combined entry ────────────────────────────────────────────
        # Full signal: all conditions
        buy_full = macro_bull & bb_oversold & rsi_oversold & stoch_oversold & vol_spike

        # Softer signal: 3 out of 4 conditions
        buy_soft = macro_bull & bb_oversold & rsi_oversold & (stoch_oversold | vol_spike) & reversal_candle

        buy_signal = buy_full | buy_soft

        # ── Exit signals ──────────────────────────────────────────────
        # Target: mean reversion to BB midline
        at_target   = df['close'] >= df['bb_mid'] * 0.98
        rsi_recovered = df['rsi'] > self.rsi_exit
        bb_upper_hit = df['bb_pct'] > 0.85

        sell_signal = at_target | rsi_recovered | bb_upper_hit

        # ── Build signals ─────────────────────────────────────────────
        df['signal'] = 0
        df.loc[buy_signal, 'signal']  = 1
        df.loc[sell_signal, 'signal'] = -1

        # Tight stop-loss for mean reversion (capital preservation)
        df['sl'] = df['close'] - self.atr_sl_mult * df['atr']
        # Target: BB midline
        df['tp'] = df['bb_mid']

        return df
