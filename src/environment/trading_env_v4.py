"""
trading_env_v4.py — PPO environment with reward shaping from all 3 static strategies.

What changed from v3 → v4:
─────────────────────────────────────────────────────────────────────────────
CHANGE 1 ► Observation is now 88-dim (was 76)
  Strategy features expanded from 6 → 18.
  The PPO can now SEE every condition that the 3 static strategies use:
    BB-RSI:   bb_oversold, rsi_oversold, stoch_oversold, volume_spike,
              reversal_candle, bb_at_upper, stoch_overbought, bbrsi_setup
    MACD-ADX: macd_just_crossed_up, macd_just_crossed_dn, strong_trend,
              di_bull, macd_adx_setup
    MIC:      mic_setup, ema8_34_cross
  The PPO doesn't hardcode these as rules — it learns WHEN and HOW MUCH
  to weight each signal via policy gradients.

CHANGE 2 ► BB-RSI Entry Bonus (+0.004)
  Source: bb_rsi_strategy.py — "buy oversold bounces in a bull market"
  When agent goes LONG and the full BB-RSI setup is present:
    (bb_pct < 0.15) AND (RSI < 35) AND (price > EMA200) AND (stoch < 25 OR vol spike)
  This teaches the PPO that oversold bounces in bull markets are high-quality entries.
  The magnitude (+0.004) is slightly above the EMA entry bonus (+0.003 from v3)
  because BB-RSI has a historically higher win rate (80%+ in research).

CHANGE 3 ► BB-RSI Overbought Hold Penalty (-0.003/step)
  Source: bb_rsi_strategy.py — exit at BB midline or when RSI > 55
  When agent holds LONG while bb_pct > 0.85 OR rsi_overbought:
  The mean reversion trade is DONE. Teach the PPO to exit, not overstay.
  This prevents the "bag-holding" failure mode of mean reversion strategies.

CHANGE 4 ► MACD-ADX Trend Entry Bonus (+0.003)
  Source: macd_adx_strategy.py — "enter on fresh MACD cross with strong trend"
  When agent goes LONG and macd_just_crossed_up AND ADX > 20 AND DI+ > DI-:
  Research paper (arXiv:2511.00665) shows this setup outperforms random entry in BTC.
  Teaches the PPO to time entries with momentum shifts.

CHANGE 5 ► MACD-ADX Trend Death Penalty (-0.003/step)
  Source: macd_adx_strategy.py — exit when MACD crosses down or DI- > DI+
  When agent holds LONG while macd_just_crossed_dn OR DI- > DI+:
  The trend momentum has reversed. Every step held here is a step losing money.
  This is the symmetric exit teaching to Change 4's entry teaching.

CHANGE 6 ► MIC Full Confluence Bonus (+0.005)
  Source: mic_strategy.py — "only enter when ALL indicators agree simultaneously"
  When agent goes LONG and EMA cross + MACD + RSI zone + volume ALL agree:
  This is the highest-quality setup — IEEE paper shows profit factor 3.5 on this.
  The +0.005 bonus is the LARGEST entry bonus, reflecting the highest signal quality.

CHANGE 7 ► Stochastic Overbought Hold Penalty (-0.002/step)
  Source: bb_rsi_strategy.py — uses stoch_k < 25 for entry (symmetric logic)
  When agent holds LONG while stoch_k > 80:
  The short-term oscillator says overbought. Trim or exit the position.
  This is a softer penalty than the regime penalty because stoch is faster/noisier.

Summary of all reward signals (v4):
─────────────────────────────────────────────────────────────────────────────
  Trade close:        log-return × scale (asymmetric: losses penalized 1.5×)
  Holding:            0.05 × Δ(unrealized PnL) [dense signal]
  Flat:              -0.0001/step [anti-lazy penalty]

  From v3 (kept):
  Regime penalty:    -0.005/step when long AND price < EMA200
  Trailing penalty:  -0.003/step when long AND price < EMA21
  EMA Entry bonus:   +0.003 one-time when entering on EMA8/21 cross in bull
  Churn penalty:     -0.010 when closing trade held < 12 steps
  Cooldown:           6-hour forced flat after every trade close

  From v4 (new - from 3 static strategies):
  BB-RSI Entry:      +0.004 when entering on full BB-RSI oversold setup
  BB Overbought:     -0.003/step when long AND (bb_at_upper OR rsi_overbought)
  MACD-ADX Entry:    +0.003 when entering on MACD cross + strong ADX + DI bull
  Trend Death:       -0.003/step when long AND (MACD crossed dn OR DI bearish)
  MIC Confluence:    +0.005 when entering on full MIC all-signals-agree setup
  Stoch Overbought:  -0.002/step when long AND stoch_k > 80

Place at: src/environment/trading_env_v4.py
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import SEQUENCE_LENGTH, STATE_DIM, STOP_LOSS_PCT, DEVICE_LSTM
from src.data.features_v3 import PPO_STRATEGY_FEATURES_V4


class CryptoTradingEnvV4(gym.Env):
    """
    PPO trading environment with reward shaping from all 3 static strategies.

    Observation (88-dim):
        [0:64]  BiLSTM market state — compressed 60-candle pattern
        [64:70] Account features:
                  [is_flat, is_long, is_short, unrealized_pnl,
                   balance_ratio, steps_in_pos]
        [70:88] Strategy features (18 signals from all 3 static strategies):
                  See PPO_STRATEGY_FEATURES_V4 in features_v3.py

    Actions (position targets):
        0 = Go Flat   (close any position)
        1 = Go Long   (enter/hold long)
        2 = Go Short  (enter/hold short)

    Reward shaping philosophy:
        Each static strategy encodes human expertise about WHEN to enter/exit.
        Instead of hardcoding those rules as strict filters (which makes the
        policy brittle), we express them as soft reward signals. The PPO
        learns to weight these signals based on their actual profitability
        in the training environment — if BB-RSI setups are genuinely more
        profitable, the agent will learn to seek them without us forcing it.
    """

    # ── From v3 (unchanged) ───────────────────────────────────────────────────
    PROFIT_SCALE     = 1.0
    LOSS_SCALE       = 1.3
    DENSE_SCALE      = 0.07
    FLAT_PENALTY     = -1e-4
    REGIME_PENALTY   = -0.005   # /step: long in bear (price < EMA200)
    TRAILING_PENALTY = -0.003   # /step: long below EMA21 (trailing stop breach)
    ENTRY_BONUS      = 0.003    # one-time: entering on EMA8/21 cross in bull market
    CHURN_PENALTY    = -0.006   # closing trade held < MIN_HOLD_STEPS
    MAX_HOLD_STEPS   = 200
    MIN_HOLD_STEPS   = 12       # 12h minimum hold or churn penalty
    COOLDOWN_STEPS   = 4        # 4h forced flat after close
    DRAWDOWN_PENALTY = -0.002   # per-step penalty scaled by current drawdown

    # ── New v4 rewards (from 3 static strategies) ─────────────────────────────

    # BB-RSI Mean Reversion (bb_rsi_strategy.py)
    # +0.004 entry: oversold bounce setup in bull market (80%+ win rate in research)
    # -0.003 hold:  mean reversion complete — stop holding at upper BB / overbought
    BBRSI_ENTRY_BONUS       = 0.004
    BBRSI_OVERBOUGHT_PENALTY = -0.003   # /step

    # MACD-ADX Trend Following (macd_adx_strategy.py)
    # +0.003 entry: fresh MACD cross + confirmed strong trend (arXiv:2511.00665)
    # -0.003 hold:  trend momentum reversed — MACD flipped or DI bearish
    MACD_ADX_ENTRY_BONUS    = 0.003
    TREND_DEATH_PENALTY     = -0.003   # /step

    # MIC Full Confluence (mic_strategy.py)
    # +0.005 entry: ALL indicators agree simultaneously (profit factor 3.5 in IEEE paper)
    # This is the HIGHEST entry bonus because confluence signals are highest quality
    MIC_ENTRY_BONUS         = 0.005

    # Stochastic Overbought (symmetric to BB-RSI's stoch_oversold entry condition)
    # -0.002 hold:  short-term oscillator says overbought — scale back
    STOCH_OB_PENALTY        = -0.002   # /step

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        initial_balance: float = 10_000.0,
        transaction_fee: float = 0.001,
        use_lstm_features: bool = True,
        lstm_model=None,
        render_mode=None,
    ):
        super().__init__()
        self.df              = df.reset_index(drop=True)
        self.feature_cols    = feature_cols
        self.initial_balance = initial_balance
        self.fee             = transaction_fee
        self.use_lstm        = use_lstm_features and lstm_model is not None
        self.lstm_model      = lstm_model
        self.render_mode     = render_mode

        if self.use_lstm:
            self.lstm_model.to(DEVICE_LSTM)
            self.lstm_model.eval()

        self.prices   = self.df['close'].values.astype(np.float64)
        self.features = self.df[feature_cols].values.astype(np.float32)
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=1.0, neginf=-1.0)
        self.n_steps  = len(self.df)

        # ── Pre-extract v3 regime arrays (from v3, kept for compatibility) ────
        def _col(name, default):
            if name in self.df.columns:
                return self.df[name].values.astype(np.float32)
            return np.full(len(self.df), default, dtype=np.float32)

        self.price_above_200   = _col('price_above_200', 1.0)
        self.ema8_21_cross     = _col('ema8_21_cross', 0.0)
        self.price_above_ema21 = _col('price_above_ema21', 1.0)

        # ── Pre-extract v4 strategy signal arrays ─────────────────────────────
        # BB-RSI signals
        self.bbrsi_setup       = _col('bbrsi_setup', 0.0)
        self.bb_at_upper       = _col('bb_at_upper', 0.0)
        self.rsi_overbought    = _col('rsi_overbought', 0.0)
        self.stoch_overbought  = _col('stoch_overbought', 0.0)

        # MACD-ADX signals
        self.macd_just_crossed_up = _col('macd_just_crossed_up', 0.0)
        self.macd_just_crossed_dn = _col('macd_just_crossed_dn', 0.0)
        self.macd_adx_setup       = _col('macd_adx_setup', 0.0)
        self.di_bull              = _col('di_bull', 0.0)
        # strong_trend (ADX > 25) gates the DI penalty — prevents noise-firing in
        # ranging markets where DI+/DI- flip constantly (would fire ~50% of bars)
        self.strong_trend         = _col('strong_trend', 0.0)

        # MIC signals
        self.mic_setup = _col('mic_setup', 0.0)

        # ── Pre-extract observation feature arrays ────────────────────────────
        self._strat_arrays = {}
        for col in PPO_STRATEGY_FEATURES_V4:
            self._strat_arrays[col] = _col(col, 0.0)

        # ── Spaces ────────────────────────────────────────────────────────────
        self.action_space = spaces.Discrete(3)

        market_dim = STATE_DIM if self.use_lstm else len(feature_cols)
        obs_dim    = market_dim + 6 + len(PPO_STRATEGY_FEATURES_V4)  # 64+6+18 = 88
        self.market_dim = market_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ── State ─────────────────────────────────────────────────────────────
        self.current_step    = SEQUENCE_LENGTH
        self.balance         = initial_balance
        self.position        = 0
        self.entry_price     = 0.0
        self.entry_step      = 0
        self.prev_unrealized = 0.0
        self.trades          = []
        self.cooldown_until  = 0
        self.peak_equity     = initial_balance

    # ─────────────────────────────────────────────────────────────────────────
    # Gym interface
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step    = SEQUENCE_LENGTH
        self.balance         = self.initial_balance
        self.position        = 0
        self.entry_price     = 0.0
        self.entry_step      = SEQUENCE_LENGTH
        self.prev_unrealized = 0.0
        self.trades          = []
        self.cooldown_until  = 0
        self.peak_equity     = self.initial_balance
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        price  = float(self.prices[self.current_step])
        reward = 0.0
        target = {0: 0, 1: 1, 2: -1}[action]

        # Enforce cooldown — agent forced flat after every trade close
        in_cooldown = self.current_step < self.cooldown_until
        if in_cooldown and self.position == 0 and target != 0:
            target = 0

        if target == self.position:
            reward = self._hold_reward(price)
        elif target == 0:
            reward = self._close_position(price, reason='signal')
        elif self.position == 0:
            # Opening new position — add all entry bonuses
            reward  = self._open_reward_v3(price, direction=target)    # EMA bonus (v3)
            reward += self._open_reward_v4(price, direction=target)    # Strategy bonuses (v4)
            self._open_position(price, direction=target)
            self.prev_unrealized = 0.0
        else:
            # Flip: close existing + open opposite
            reward += self._close_position(price, reason='flip')
            reward += self._open_reward_v3(price, direction=target)
            reward += self._open_reward_v4(price, direction=target)
            self._open_position(price, direction=target)
            self.prev_unrealized = 0.0

        if self.position != 0:
            reward += self._check_stop_loss(price)
            reward += self._regime_reward_v3()   # regime + trailing (from v3)
            reward += self._hold_penalty_v4()    # strategy hold penalties (from v4)

        # Drawdown penalty based on current equity vs peak equity
        if self.position == 0:
            equity = self.balance
        elif self.position == 1:
            equity = self.balance * (1.0 + (price - self.entry_price) / self.entry_price)
        else:
            equity = self.balance * (1.0 + (self.entry_price - price) / self.entry_price)
        if equity > self.peak_equity:
            self.peak_equity = equity
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity
            drawdown = float(np.clip(drawdown, 0.0, 0.5))
            reward += self.DRAWDOWN_PENALTY * drawdown

        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated  = self.balance <= self.initial_balance * 0.5

        obs  = self._get_obs()
        info = self._get_info()

        if terminated or truncated:
            if self.position != 0:
                ep = float(self.prices[self.current_step - 1])
                self._close_position(ep, reason='eod')
            info['final_balance'] = float(self.balance)
            info['total_return']  = float(self.balance / self.initial_balance - 1.0)
            info['n_trades']      = len(self.trades)
            if self.trades:
                avg_hold = float(np.mean([t['hold_duration'] for t in self.trades]))
            else:
                avg_hold = 0.0
            info['avg_hold'] = avg_hold

        return obs, float(reward), terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────────────
    # Reward helpers — v3 (carried over unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    def _open_reward_v3(self, price: float, direction: int) -> float:
        """
        (From v3) Bonus for entering long on the EMA8/21 cross in a bull market.
        Source: optimized_trend.py — the strategy that drove the +7.96% baseline ROI.
        """
        if direction != 1:
            return 0.0
        i = self.current_step
        perfect_entry = (
            self.price_above_200[i] == 1 and
            self.ema8_21_cross[i] == 1
        )
        return self.ENTRY_BONUS if perfect_entry else 0.0

    def _regime_reward_v3(self) -> float:
        """
        (From v3) Per-step penalties for:
          1. Being long in a bear market (price < EMA200)
          2. Price below EMA21 trailing stop
        """
        if self.position != 1:
            return 0.0
        i = self.current_step - 1
        reward = 0.0
        if self.price_above_200[i] == 0:
            reward += self.REGIME_PENALTY
        if self.price_above_ema21[i] == 0:
            reward += self.TRAILING_PENALTY
        return reward

    # ─────────────────────────────────────────────────────────────────────────
    # Reward helpers — v4 (new, from 3 static strategies)
    # ─────────────────────────────────────────────────────────────────────────

    def _open_reward_v4(self, price: float, direction: int) -> float:
        """
        NEW v4 entry bonuses from the 3 static strategies.

        Priority order (additive — multiple bonuses can stack):
          1. MIC confluence (+0.005)  — highest quality, all indicators agree
          2. BB-RSI setup (+0.004)    — oversold bounce in bull market
          3. MACD-ADX setup (+0.003)  — fresh MACD cross with confirmed trend

        Why additive? When multiple strategies agree on the same entry
        (e.g., both MIC and MACD-ADX fire), that's even higher confluence.
        Maximum possible entry bonus: 0.005 + 0.004 + 0.003 + 0.003 = 0.015

        For short entries (direction == -1): no bonuses are given because
        our static strategies are long-only. The regime penalty already
        discourages bad short entries.
        """
        if direction != 1:
            return 0.0

        i = self.current_step
        reward = 0.0

        # ── MIC Full Confluence (+0.005) ──────────────────────────────────────
        # Source: mic_strategy.py — "ALL indicators must agree simultaneously"
        # IEEE research shows profit factor 3.5 on this setup — highest quality.
        if self.mic_setup[i] == 1:
            reward += self.MIC_ENTRY_BONUS

        # ── BB-RSI Oversold Bounce (+0.004) ───────────────────────────────────
        # Source: bb_rsi_strategy.py — 80%+ win rate in ClucMay72018 research.
        # Only fires in bull market (price_above_200 is embedded in bbrsi_setup).
        if self.bbrsi_setup[i] == 1:
            reward += self.BBRSI_ENTRY_BONUS

        # ── MACD-ADX Trend Entry (+0.003) ─────────────────────────────────────
        # Source: macd_adx_strategy.py — arXiv:2511.00665, optimal BTC params.
        if self.macd_adx_setup[i] == 1:
            reward += self.MACD_ADX_ENTRY_BONUS

        return reward

    def _hold_penalty_v4(self) -> float:
        """
        NEW v4 per-step penalties while holding a position.

        Only applies when LONG (position == 1). The penalties teach the PPO
        to EXIT at the same time the static strategies would exit:

        1. BB-RSI Overbought (-0.003/step):
           Source: bb_rsi_strategy.py — "exit at BB midline or when RSI > 55"
           When bb_pct > 0.85 OR rsi_overbought: mean reversion is complete.
           Every step held beyond this point erodes the profit.

        2. MACD-ADX Trend Death (-0.003/step):
           Source: macd_adx_strategy.py — "sell when MACD crosses down or DI- > DI+"
           When MACD just crossed down OR DI is now bearish: the trend is dead.
           Exit immediately instead of riding the drawdown.

        3. Stochastic Overbought (-0.002/step):
           Source: bb_rsi_strategy.py — stoch < 25 for entry → stoch > 80 to exit.
           Short-term oscillator says extended. Softer signal than the others.
        """
        if self.position != 1:
            return 0.0

        i = self.current_step - 1
        reward = 0.0

        # ── BB-RSI: overbought — mean reversion trade is done ─────────────────
        if self.bb_at_upper[i] == 1 or self.rsi_overbought[i] == 1:
            reward += self.BBRSI_OVERBOUGHT_PENALTY

        # ── MACD-ADX: trend momentum reversed — exit the trend trade ──────────
        # macd_just_crossed_dn: clean 1-candle event, always meaningful.
        # di_bull==0 gated by strong_trend: without the ADX gate, DI flips in
        # ranging markets ~50% of bars → constant reward noise → kills
        # explained_variance. Only penalize when ADX>25 confirms a genuinely
        # strong trend has actually reversed bearish, not just random chop.
        di_reversed = (self.di_bull[i] == 0 and self.strong_trend[i] == 1)
        if self.macd_just_crossed_dn[i] == 1 or di_reversed:
            reward += self.TREND_DEATH_PENALTY

        # ── Stochastic: short-term overbought ─────────────────────────────────
        if self.stoch_overbought[i] == 1:
            reward += self.STOCH_OB_PENALTY

        return reward

    # ─────────────────────────────────────────────────────────────────────────
    # Core position management (unchanged from v3)
    # ─────────────────────────────────────────────────────────────────────────

    def _open_position(self, price: float, direction: int):
        self.position    = direction
        self.entry_price = price * (1 + direction * self.fee)
        self.entry_step  = self.current_step

    def _close_position(self, price: float, reason: str = 'signal') -> float:
        if self.position == 0:
            return 0.0
        if self.position == 1:
            pnl_pct = (price - self.entry_price) / self.entry_price - self.fee
        else:
            pnl_pct = (self.entry_price - price) / self.entry_price - self.fee

        old_balance  = self.balance
        self.balance = self.balance * (1.0 + pnl_pct)
        log_return   = np.log(max(self.balance, 1e-8) / max(old_balance, 1e-8))

        # Asymmetric loss scale: losses penalized 1.5× to teach caution
        reward = self.PROFIT_SCALE * log_return if log_return >= 0 \
                 else self.LOSS_SCALE * log_return

        # Anti-churn: penalize short-lived trades (< 12 steps = 12 hours)
        hold_duration = self.current_step - self.entry_step
        if hold_duration < self.MIN_HOLD_STEPS and reason != 'stop_loss':
            reward += self.CHURN_PENALTY

        self.trades.append({
            'direction':    self.position,
            'entry_price':  self.entry_price,
            'exit_price':   price,
            'pnl_pct':      pnl_pct,
            'log_return':   log_return,
            'reason':       reason,
            'hold_duration': hold_duration,
        })
        self.position        = 0
        self.entry_price     = 0.0
        self.prev_unrealized = 0.0
        self.cooldown_until  = self.current_step + self.COOLDOWN_STEPS
        return reward

    def _hold_reward(self, price: float) -> float:
        if self.position == 0:
            return self.FLAT_PENALTY
        if self.position == 1:
            unrealized = (price - self.entry_price) / self.entry_price
        else:
            unrealized = (self.entry_price - price) / self.entry_price
        delta = unrealized - self.prev_unrealized
        self.prev_unrealized = unrealized
        return self.DENSE_SCALE * delta

    def _check_stop_loss(self, price: float) -> float:
        if self.position == 0:
            return 0.0
        pnl = (price - self.entry_price) / self.entry_price \
              if self.position == 1 else (self.entry_price - price) / self.entry_price
        if pnl < -STOP_LOSS_PCT:
            return self._close_position(price, reason='stop_loss')
        return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Observation
    # ─────────────────────────────────────────────────────────────────────────

    def _get_market_state(self) -> np.ndarray:
        if self.use_lstm:
            import torch
            start = max(0, self.current_step - SEQUENCE_LENGTH)
            seq   = self.features[start:self.current_step]
            if len(seq) < SEQUENCE_LENGTH:
                pad = np.zeros((SEQUENCE_LENGTH - len(seq), seq.shape[1]), np.float32)
                seq = np.vstack([pad, seq])
            device = next(self.lstm_model.parameters()).device
            x      = torch.as_tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            return np.asarray(
                self.lstm_model.extract_features(x).squeeze(), dtype=np.float32
            )
        return self.features[self.current_step].copy()

    def _get_obs(self) -> np.ndarray:
        market_state = self._get_market_state()   # [0:64] BiLSTM

        # [64:70] Account state
        is_flat  = float(self.position == 0)
        is_long  = float(self.position == 1)
        is_short = float(self.position == -1)
        if self.position == 0:
            unrealized = 0.0
        else:
            price = float(self.prices[self.current_step])
            unrealized = (price - self.entry_price) / max(self.entry_price, 1e-8) \
                         if self.position == 1 \
                         else (self.entry_price - price) / max(self.entry_price, 1e-8)
        unrealized    = float(np.clip(unrealized, -0.5, 0.5))
        balance_ratio = float(np.clip(self.balance / self.initial_balance - 1.0, -2, 2))
        steps_held    = float(np.clip(
            (self.current_step - self.entry_step) / self.MAX_HOLD_STEPS, 0, 1
        ))
        account_state = np.array(
            [is_flat, is_long, is_short, unrealized, balance_ratio, steps_held],
            dtype=np.float32
        )

        # [70:88] Strategy state — 18 features from all 3 static strategies
        i = min(self.current_step, self.n_steps - 1)
        strategy_state = np.array(
            [self._strat_arrays[col][i] for col in PPO_STRATEGY_FEATURES_V4],
            dtype=np.float32
        )

        return np.concatenate([market_state, account_state, strategy_state])

    def _get_info(self) -> dict:
        price = float(self.prices[min(self.current_step, self.n_steps - 1)])
        return {
            'step':         self.current_step,
            'balance':      float(self.balance),
            'position':     self.position,
            'steps_in_pos': float(max(0, self.current_step - self.entry_step)),
            'current_price': price,
            'n_trades':     len(self.trades),
            'total_return': float(self.balance / self.initial_balance - 1.0),
        }

    def get_trade_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

    def render(self):
        if self.render_mode == 'human':
            i   = self._get_info()
            pos = {0: 'FLAT', 1: 'LONG', -1: 'SHORT'}[self.position]
            print(
                f"Step {i['step']:5d} | {i['current_price']:>10.2f} | "
                f"{pos:5s} | {i['balance']:>10.2f} | {i['total_return']*100:>+6.2f}%"
            )
