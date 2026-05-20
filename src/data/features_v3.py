"""
features_v3.py — Extends features_v2.py with signals from all 3 static strategies.

What's new vs v2:
─────────────────────────────────────────────────────────────────────────────
FROM BB-RSI STRATEGY (bb_rsi_strategy.py):
  bb_oversold       → price in lower 15% of Bollinger Band  (bb_pct < 0.15)
  bb_at_upper       → price at upper BB — overbought exit signal (bb_pct > 0.85)
  volume_spike      → real capitulation / participation spike (vol_ratio > 1.3)
  reversal_candle   → bullish reversal: close > open AND close > prev close
  stoch_overbought  → stoch_k > 80 (BBRsi uses stoch<25 for entry; we also need
                       the symmetric exit condition for PPO hold-penalty)
  bbrsi_setup       → composite: all BB-RSI entry conditions met simultaneously

FROM MACD-ADX STRATEGY (macd_adx_strategy.py):
  macd_just_crossed_up  → MACD freshly crossed above signal (momentum shift +)
  macd_just_crossed_dn  → MACD freshly crossed below signal (momentum shift −)
  macd_adx_setup        → composite: MACD cross + strong trend + DI bullish

FROM MIC STRATEGY (mic_strategy.py):
  mic_setup         → composite: EMA cross + MACD hist rising + RSI neutral +
                       volume confirm — the "all signals agree" condition

These are fed into PPO_STRATEGY_FEATURES_V4 so the PPO agent can SEE every
condition that its human-coded predecessors used to make buy/sell decisions.
The PPO doesn't hardcode these as rules — it learns WHEN they are reliable
and weights them dynamically via the policy gradient.

Place at: src/data/features_v3.py
"""

import numpy as np
import pandas as pd
from src.data.features_v2 import build_features, BILSTM_FEATURES


# ─── Re-export BILSTM_FEATURES unchanged ─────────────────────────────────────
# The BiLSTM input stays the same — we only add new features to the PPO layer.
BILSTM_FEATURES = BILSTM_FEATURES


def build_features_v3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full v2 feature pipeline then appends v3 strategy signals.

    Call this instead of build_features() everywhere in v3 training.

    Args:
        df: Raw OHLCV DataFrame (must have open, high, low, close, volume columns)

    Returns:
        DataFrame with all v2 features + new v3 strategy signal columns.
    """
    # ── 1. Run v2 pipeline first ───────────────────────────────────────────────
    df = build_features(df)

    c  = df['close']
    o  = df['open']

    # ═════════════════════════════════════════════════════════════════════════
    # BB-RSI STRATEGY SIGNALS
    # Source: strategies/bb_rsi_strategy.py
    # Logic: buy oversold bounces in a macro bull market
    # ═════════════════════════════════════════════════════════════════════════

    # Price is in the lower 15% of the Bollinger Band — setup zone
    # (v2 already has bb_pct; we threshold it here)
    df['bb_oversold'] = (df['bb_pct'] < 0.15).astype(int)

    # Price is at the upper 15% of the BB — this is where bb_rsi_strategy EXITS
    # We use this as a long-hold penalty trigger for PPO
    df['bb_at_upper'] = (df['bb_pct'] > 0.85).astype(int)

    # Volume spike: > 1.3× 20-period average — confirms real capitulation/buying
    # bb_rsi_strategy uses vol_ratio > 1.3 as entry confirmation
    df['volume_spike'] = (df['volume_ratio'] > 1.3).astype(int)

    # Bullish reversal candle: close above open AND above previous close
    # bb_rsi_strategy's "softer signal" uses this as a final filter
    df['reversal_candle'] = (
        (c > o) & (c > c.shift(1))
    ).astype(int)

    # Stochastic overbought — symmetric exit condition to stoch_oversold
    # When stoch_k > 80 while long, the BB-RSI trade is over; exit zone
    df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)

    # ── BB-RSI composite setup flag ────────────────────────────────────────────
    # True when: bb_oversold AND rsi_oversold AND macro_bull AND
    #            (stoch_oversold OR volume_spike)
    # This is the exact "full signal" from bb_rsi_strategy.generate_signals()
    bbrsi_full = (
        (df['bb_oversold'] == 1) &
        (df['rsi_oversold'] == 1) &
        (df['price_above_200'] == 1) &
        ((df['stoch_oversold'] == 1) | (df['volume_spike'] == 1))
    )
    # Softer version: 3/4 conditions + reversal candle
    bbrsi_soft = (
        (df['bb_oversold'] == 1) &
        (df['rsi_oversold'] == 1) &
        (df['price_above_200'] == 1) &
        (df['reversal_candle'] == 1)
    )
    df['bbrsi_setup'] = (bbrsi_full | bbrsi_soft).astype(int)

    # ═════════════════════════════════════════════════════════════════════════
    # MACD-ADX STRATEGY SIGNALS
    # Source: strategies/macd_adx_strategy.py
    # Logic: enter on fresh MACD cross when trend is confirmed strong
    # Uses research-optimal MACD(17,21,15) — already in v2 as macd_opt
    # ═════════════════════════════════════════════════════════════════════════

    # MACD freshly crossed ABOVE signal — momentum shift positive
    # macd_adx_strategy.py: "macd_just_crossed" → buy signal
    df['macd_just_crossed_up'] = (
        (df['macd_opt'] > df['macd_opt_signal']) &
        (df['macd_opt'].shift(1) <= df['macd_opt_signal'].shift(1))
    ).astype(int)

    # MACD freshly crossed BELOW signal — momentum dying
    # macd_adx_strategy.py: "macd_cross_down" → sell signal
    # PPO should EXIT when this fires while holding long
    df['macd_just_crossed_dn'] = (
        (df['macd_opt'] < df['macd_opt_signal']) &
        (df['macd_opt'].shift(1) >= df['macd_opt_signal'].shift(1))
    ).astype(int)

    # ── MACD-ADX composite setup flag ─────────────────────────────────────────
    # True when: fresh MACD cross up + strong trend (ADX>20) + DI bullish
    # This is the "buy_signal" entry from macd_adx_strategy
    df['macd_adx_setup'] = (
        (df['macd_just_crossed_up'] == 1) &
        (df['adx'] >= 20) &
        (df['di_bull'] == 1)
    ).astype(int)

    # ── ADX trend death signal ─────────────────────────────────────────────────
    # When DI- > DI+ the trend has flipped bearish — MACD-ADX strategy exits
    # This is a continuous condition (not just a cross) so we flag it
    df['di_bearish'] = (df['di_minus'] > df['di_plus']).astype(int)

    # ═════════════════════════════════════════════════════════════════════════
    # MIC (MULTI-INDICATOR CONFLUENCE) STRATEGY SIGNALS
    # Source: strategies/mic_strategy.py
    # Logic: only enter when EMA trend + MACD momentum + RSI + Volume ALL agree
    # ═════════════════════════════════════════════════════════════════════════

    # ── MIC composite setup flag ───────────────────────────────────────────────
    # Exactly mirrors mic_strategy.py buy_signal conditions:
    #   ema8_34_cross == 1  (EMA8 > EMA34, bullish trend)
    #   price_above_200     (macro uptrend)
    #   macd_hist > 0 AND rising  (momentum confirms)
    #   RSI in [45, 65]     (not overbought, positive zone)
    #   volume_ratio > 1.0  (volume participation)
    mic_base = (
        (df['ema8_34_cross'] == 1) &
        (df['price_above_200'] == 1) &
        (df['macd_hist'] > 0) &
        (df['macd_hist_rising'] == 1) &
        (df['rsi_neutral'] == 1) &           # v2 already defines rsi_neutral = RSI 45-65
        (df['volume_ratio'] >= 1.0)
    )
    # Strong trend continuation — additional MIC condition from mic_strategy.py
    mic_strong = (
        (df['ema8_34_cross'] == 1) &
        (df['price_above_200'] == 1) &
        (df['macd_hist'] > 0) &
        (df['macd_hist_rising'] == 1) &
        (df['adx'] > 30) &
        (df['volume_ratio'] >= 1.0)
    )
    df['mic_setup'] = (mic_base | mic_strong).astype(int)

    # ── MIC exit flag ──────────────────────────────────────────────────────────
    # mic_strategy exits when EMA cross reverses OR RSI overbought OR MACD flips
    # We flag this explicitly so PPO can be penalized for ignoring it
    df['mic_exit'] = (
        (df['ema8_34_cross'] == 0) |
        (df['rsi'] > 70) |
        (df['macd_hist'] < 0)
    ).astype(int)

    print(
        f"[features_v3] New strategy signals added:\n"
        f"  BB-RSI  → bb_oversold={df['bb_oversold'].sum():,}  "
        f"bb_at_upper={df['bb_at_upper'].sum():,}  "
        f"bbrsi_setup={df['bbrsi_setup'].sum():,} rows\n"
        f"  MACD-ADX → macd_just_crossed_up={df['macd_just_crossed_up'].sum():,}  "
        f"macd_just_crossed_dn={df['macd_just_crossed_dn'].sum():,}  "
        f"macd_adx_setup={df['macd_adx_setup'].sum():,} rows\n"
        f"  MIC     → mic_setup={df['mic_setup'].sum():,}  "
        f"mic_exit={df['mic_exit'].sum():,} rows"
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PPO_STRATEGY_FEATURES_V4
# ─────────────────────────────────────────────────────────────────────────────
# Extended from v3's 6 features → 18 features.
# Obs space: [64 BiLSTM] + [6 account] + [18 strategy] = 88-dim total.
#
# Each group maps to a static strategy:
#   [REGIME]   — shared prerequisite across all 3 strategies
#   [BB-RSI]   — from bb_rsi_strategy.py
#   [MACD-ADX] — from macd_adx_strategy.py
#   [MIC]      — from mic_strategy.py
#   [NORM]     — continuous normalized features (allow fine-grained weighting)
# ─────────────────────────────────────────────────────────────────────────────

PPO_STRATEGY_FEATURES_V4 = [
    # ── REGIME (used by all 3 strategies as prerequisite) ────────────────────
    'price_above_200',        # [70] master bear/bull filter — never trade against macro
    'ema8_21_cross',          # [71] fresh EMA8/21 momentum cross (optimized_trend entry)
    'price_above_ema21',      # [72] trailing stop signal — exit when price drops below EMA21

    # ── BB-RSI MEAN REVERSION (from bb_rsi_strategy.py) ─────────────────────
    'bb_oversold',            # [73] price in lower 15% of BB — setup zone
    'rsi_oversold',           # [74] RSI < 35 — oversold, panic selling
    'stoch_oversold',         # [75] stoch_k < 25 — confirms oversold condition
    'volume_spike',           # [76] vol_ratio > 1.3 — real capitulation / breakout
    'reversal_candle',        # [77] bullish reversal candle (close > open & prev close)
    'bb_at_upper',            # [78] bb_pct > 0.85 — mean reversion target reached, EXIT
    'stoch_overbought',       # [79] stoch_k > 80 — overbought, BB-RSI exit zone
    'bbrsi_setup',            # [80] COMPOSITE: full BB-RSI entry conditions met

    # ── MACD-ADX TREND FOLLOWING (from macd_adx_strategy.py) ────────────────
    'macd_just_crossed_up',   # [81] MACD(17,21,15) freshly crossed up — trend entry
    'macd_just_crossed_dn',   # [82] MACD freshly crossed down — trend reversal EXIT
    'strong_trend',           # [83] ADX > 25 — trend is real, not choppy sideways
    'di_bull',                # [84] DI+ > DI- — directional indicator bullish
    'macd_adx_setup',         # [85] COMPOSITE: fresh cross + strong ADX + DI bullish

    # ── MIC CONFLUENCE (from mic_strategy.py) ────────────────────────────────
    'mic_setup',              # [86] COMPOSITE: EMA + MACD + RSI + volume all agree
    'ema8_34_cross',          # [87] EMA8 > EMA34 (MIC's trend filter, wider than EMA21)
]

# Sanity: 18 features + 64 BiLSTM + 6 account = 88-dim observation
assert len(PPO_STRATEGY_FEATURES_V4) == 18, \
    f"Expected 18 strategy features, got {len(PPO_STRATEGY_FEATURES_V4)}"
