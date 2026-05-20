"""
trading_env_v3.py — PPO environment built on the +7.96% ROI strategy findings.

What changed from v2 → v3:
─────────────────────────────────────────────────────────────────
CHANGE 1 ► Observation now 76-dim (was 70)
  Added 6 strategy features directly to PPO observation:
  [price_above_200, ema8_21_cross, price_above_ema21, adx_norm, rsi_norm, bb_pct_norm]
  The PPO can now SEE the regime filter and entry signal explicitly.

CHANGE 2 ► Regime penalty in reward
  When long AND price_above_200 = 0 → extra -0.005/step
  This directly teaches the PPO the #1 rule from optimized_trend.py:
  "never go long in a bear market"

CHANGE 3 ► Trailing stop signal in reward
  When long AND price_above_ema21 = 0 → extra -0.003/step
  This teaches the PPO to exit when price drops below EMA21
  (the exact exit rule that kept max drawdown at 5.96%)

CHANGE 4 ► Entry quality bonus
  When ema8_21_cross=1 AND price_above_200=1 AND agent goes long → +0.003
  Rewards the PPO for entering on the exact setup that produced +7.96%

Place at: src/environment/trading_env_v3.py
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import SEQUENCE_LENGTH, STATE_DIM, STOP_LOSS_PCT, DEVICE_LSTM
from src.data.features_v2 import PPO_STRATEGY_FEATURES


class CryptoTradingEnvV3(gym.Env):
    """
    Observation (76-dim):
        [0:64]  BiLSTM market state
        [64:70] Account features: [is_flat, is_long, is_short,
                                    unrealized_pnl, balance_ratio, steps_in_pos]
        [70:76] Strategy features: [price_above_200, ema8_21_cross,
                                     price_above_ema21, adx_norm, rsi_norm, bb_pct_norm]

    Actions (position targets):
        0 = Go Flat   (close any position)
        1 = Go Long   (enter/hold long)
        2 = Go Short  (enter/hold short)

    Reward:
        Trade close:   log(new_balance / old_balance) × scale
                       losses: 1.5× multiplier (asymmetric — teaches caution)
        Holding:       0.05 × Δ(unrealized_pnl)  [dense signal]
        Flat:         -0.0001/step               [anti-lazy penalty]
        + Regime penalty:    -0.005/step when long in bear market
        + Trailing penalty:  -0.003/step when long but price < EMA21
        + Entry bonus:       +0.003 when entering on perfect setup
        + Churn penalty:     -0.01 when closing trade held < MIN_HOLD_STEPS
        + Cooldown:          Agent forced flat for COOLDOWN_STEPS after each close
    """

    PROFIT_SCALE    = 1.0
    LOSS_SCALE      = 1.5
    DENSE_SCALE     = 0.05
    FLAT_PENALTY    = -1e-4
    REGIME_PENALTY  = -0.005   # per step: long in bear market
    TRAILING_PENALTY= -0.003   # per step: long below EMA21
    ENTRY_BONUS     = 0.003    # one-time: entering on perfect setup
    CHURN_PENALTY   = -0.01    # closing trade held < MIN_HOLD_STEPS
    MAX_HOLD_STEPS  = 200
    MIN_HOLD_STEPS  = 12       # must hold at least 12h or get penalized
    COOLDOWN_STEPS  = 6        # forced flat for 6h after closing a trade

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

        # Strategy signal arrays (pre-extracted for speed)
        self.price_above_200   = self.df['price_above_200'].values.astype(np.float32) \
            if 'price_above_200' in self.df.columns else np.ones(len(self.df), dtype=np.float32)
        self.ema8_21_cross     = self.df['ema8_21_cross'].values.astype(np.float32) \
            if 'ema8_21_cross' in self.df.columns else np.zeros(len(self.df), dtype=np.float32)
        self.price_above_ema21 = self.df['price_above_ema21'].values.astype(np.float32) \
            if 'price_above_ema21' in self.df.columns else np.ones(len(self.df), dtype=np.float32)

        # Strategy feature arrays for observation
        self._strat_arrays = {}
        for col in PPO_STRATEGY_FEATURES:
            if col in self.df.columns:
                self._strat_arrays[col] = self.df[col].values.astype(np.float32)
            else:
                self._strat_arrays[col] = np.zeros(len(self.df), dtype=np.float32)

        self.action_space = spaces.Discrete(3)

        market_dim = STATE_DIM if self.use_lstm else len(feature_cols)
        obs_dim    = market_dim + 6 + len(PPO_STRATEGY_FEATURES)  # 64+6+6 = 76
        self.market_dim = market_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.current_step    = SEQUENCE_LENGTH
        self.balance         = initial_balance
        self.position        = 0
        self.entry_price     = 0.0
        self.entry_step      = 0
        self.prev_unrealized = 0.0
        self.trades          = []
        self.cooldown_until  = 0          # step when cooldown ends

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step    = SEQUENCE_LENGTH
        self.balance         = self.initial_balance
        self.position        = 0
        self.entry_price     = 0.0
        self.entry_step      = SEQUENCE_LENGTH
        self.prev_unrealized = 0.0
        self.trades          = []
        self.cooldown_until  = 0          # step when cooldown ends
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        price   = float(self.prices[self.current_step])
        reward  = 0.0
        target  = {0: 0, 1: 1, 2: -1}[action]

        # ── Anti-overtrading: enforce cooldown after trade close ──────────
        in_cooldown = self.current_step < self.cooldown_until
        if in_cooldown and self.position == 0 and target != 0:
            # Agent wants to open during cooldown → force flat
            target = 0

        if target == self.position:
            reward = self._hold_reward(price)
        elif target == 0:
            reward = self._close_position(price, reason='signal')
        elif self.position == 0:
            reward = self._open_reward(price, direction=target)
            self._open_position(price, direction=target)
            self.prev_unrealized = 0.0
        else:
            reward += self._close_position(price, reason='flip')
            reward += self._open_reward(price, direction=target)
            self._open_position(price, direction=target)
            self.prev_unrealized = 0.0

        if self.position != 0:
            reward += self._check_stop_loss(price)
            reward += self._regime_reward()

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

        return obs, float(reward), terminated, truncated, info

    # ── Reward helpers ────────────────────────────────────────────────────────

    def _open_reward(self, price: float, direction: int) -> float:
        """Bonus for entering on the exact +7.96% setup."""
        if direction != 1:
            return 0.0
        i = self.current_step
        perfect_entry = (
            self.price_above_200[i] == 1 and
            self.ema8_21_cross[i] == 1
        )
        return self.ENTRY_BONUS if perfect_entry else 0.0

    def _regime_reward(self) -> float:
        """
        Penalty for being long in a bear market (price < EMA200)
        or when price has fallen below EMA21 (trailing stop breach).
        These are the two exit rules from optimized_trend.py.
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
        reward       = self.PROFIT_SCALE * log_return if log_return >= 0 \
                       else self.LOSS_SCALE * log_return

        # ── Anti-overtrading: churn penalty for short-lived trades ────────
        hold_duration = self.current_step - self.entry_step
        if hold_duration < self.MIN_HOLD_STEPS and reason != 'stop_loss':
            reward += self.CHURN_PENALTY  # penalize quick flips

        self.trades.append({
            'direction': self.position, 'entry_price': self.entry_price,
            'exit_price': price, 'pnl_pct': pnl_pct,
            'log_return': log_return, 'reason': reason,
            'hold_duration': hold_duration,
        })
        self.position        = 0
        self.entry_price     = 0.0
        self.prev_unrealized = 0.0
        # Start cooldown — agent must stay flat for COOLDOWN_STEPS
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

    # ── Observation ───────────────────────────────────────────────────────────

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
        market_state = self._get_market_state()

        # Account state (6 features)
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

        # Strategy state (6 features from +7.96% strategy)
        i = min(self.current_step, self.n_steps - 1)
        strategy_state = np.array(
            [self._strat_arrays[col][i] for col in PPO_STRATEGY_FEATURES],
            dtype=np.float32
        )

        return np.concatenate([market_state, account_state, strategy_state])

    def _get_info(self) -> dict:
        price = float(self.prices[min(self.current_step, self.n_steps - 1)])
        return {
            'step': self.current_step, 'balance': float(self.balance),
            'position': self.position, 'current_price': price,
            'n_trades': len(self.trades),
            'total_return': float(self.balance / self.initial_balance - 1.0),
        }

    def get_trade_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

    def render(self):
        if self.render_mode == 'human':
            i = self._get_info()
            pos = {0: 'FLAT', 1: 'LONG', -1: 'SHORT'}[self.position]
            print(f"Step {i['step']:5d} | {i['current_price']:>10.2f} | "
                  f"{pos:5s} | {i['balance']:>10.2f} | {i['total_return']*100:>+6.2f}%")
