"""
CryptoTradingEnv v2 — Fixed environment for Bi-LSTM + PPO.

WHY THIS FILE EXISTS — bugs found in trading_env.py:
──────────────────────────────────────────────────────
BUG 1 ► Observation had NO position info.
  The agent received a 64-dim state from the BiLSTM but had NO idea whether
  it was currently long, short, or flat. This is like asking a chess player to
  move without showing them the board state. The agent could not learn
  position-dependent behaviour (e.g. "I am already long, don't buy again").

BUG 2 ► Reward function was miscalibrated.
  The Optuna search found: reward_profit=2.498, reward_loss=-0.623.
  This means winning trades get 4× MORE reward than losing trades get
  penalty. That teaches the agent to OVERTRADE — every trade feels like a
  great idea because wins dominate. Research (Kochliaridis 2023, Bandarupalli
  2025) shows that loss_penalty should be ≥ profit_reward for caution.

BUG 3 ► Dense unrealized reward was double-counting.
  The env added reward for BOTH (a) change in unrealized PnL and
  (b) a fixed reward_hold_winning per step. These two signals overlap and
  give contradictory gradients.

BUG 4 ► Action semantics were ambiguous.
  action=0 (Buy): if flat → enter long. if short → close short.
  The meaning of the same action changed depending on hidden state that the
  agent couldn't see (because position wasn't in the observation). This made
  the policy gradient signal extremely noisy.

FIXES IN V2:
──────────────
FIX 1 ► Observation = BiLSTM 64-dim + 6 position/account features = 70-dim
  Added: [is_flat, is_long, is_short, unrealized_pnl, balance_ratio,
          steps_in_position_normalized]

FIX 2 ► Reward = step-wise log portfolio return (research-standard).
  When a trade closes: reward = log(new_balance / old_balance)
  When holding: reward = small dense signal from unrealized PnL change
  When flat: tiny negative penalty to prevent "never trade" policy
  Loss asymmetry: loss multiplier 1.5× so the agent learns caution.

FIX 3 ► Action semantics are now POSITION TARGETS (not buy/sell commands).
  action=0: Go flat  (close any open position, do nothing if already flat)
  action=1: Go long  (open long if flat; hold if already long; flip if short)
  action=2: Go short (open short if flat; hold if already short; flip if long)
  Same action always means the same thing regardless of current state.
  The agent CAN see current state (fix 1), so it can reason about flipping.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import (
    SEQUENCE_LENGTH, STATE_DIM,
    STOP_LOSS_PCT, DEVICE_LSTM
)


class CryptoTradingEnvV2(gym.Env):
    """
    Cryptocurrency trading environment v2.

    Observation (70-dim):
        [0:64]  — BiLSTM market state (or raw features if no LSTM)
        [64]    — is_flat      (1 if no position, else 0)
        [65]    — is_long      (1 if long, else 0)
        [66]    — is_short     (1 if short, else 0)
        [67]    — unrealized_pnl  (current open trade PnL, signed, clipped ±0.2)
        [68]    — balance_ratio   (current_balance / initial_balance - 1)
        [69]    — steps_in_pos    (steps held in current position / 100, clipped 0-1)

    Actions (position targets):
        0 = Go Flat   (close position; do nothing if already flat)
        1 = Go Long   (hold if long; enter if flat; flip if short)
        2 = Go Short  (hold if short; enter if flat; flip if long)

    Reward:
        On trade close: log(new_balance / old_balance) * scale
                        loss trades get extra 1.5× penalty (asymmetry)
        While holding:  0.05 * Δ(unrealized_pnl)   [dense signal]
        While flat:    -0.0001 per step             [anti-lazy penalty]
    """

    metadata = {'render_modes': ['human']}

    # ── constants ─────────────────────────────────────────────────────────────
    PROFIT_SCALE   = 1.0    # multiplier on log-return reward for profitable close
    LOSS_SCALE     = 1.5    # asymmetric: losses hurt more (teaches caution)
    DENSE_SCALE    = 0.05   # weight on step-wise unrealized PnL change
    FLAT_PENALTY   = -1e-4  # per-step penalty for sitting flat (prevents "never trade")
    MAX_HOLD_STEPS = 200    # normalisation denominator for steps_in_position feature

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        initial_balance: float = 10_000.0,
        transaction_fee: float = 0.001,      # 0.1% Binance taker fee
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

        # Move LSTM to correct device once, here.
        if self.use_lstm:
            try:
                self.lstm_model.to(DEVICE_LSTM)
                self.lstm_model.eval()
            except Exception as e:
                raise RuntimeError(f"Cannot move BiLSTM to {DEVICE_LSTM}: {e}")

        # ── data arrays ───────────────────────────────────────────────────────
        self.prices   = self.df['close'].values.astype(np.float64)
        self.features = self.df[feature_cols].values.astype(np.float32)
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=1.0, neginf=-1.0)
        self.n_steps  = len(self.df)

        # ── spaces ────────────────────────────────────────────────────────────
        # Action: 3 position targets (flat / long / short)
        self.action_space = spaces.Discrete(3)

        # Observation: market_state + 6 account features
        market_dim = STATE_DIM if self.use_lstm else len(feature_cols)
        obs_dim    = market_dim + 6          # +6 account features
        self.market_dim = market_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        # ── episode state (initialised in reset) ──────────────────────────────
        self.current_step    = SEQUENCE_LENGTH
        self.balance         = initial_balance
        self.position        = 0      # 0=flat, 1=long, -1=short
        self.entry_price     = 0.0
        self.entry_step      = 0
        self.prev_unrealized = 0.0
        self.trades          = []

    # ──────────────────────────────────────────────────────────────────────────
    # Gym interface
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step    = SEQUENCE_LENGTH
        self.balance         = self.initial_balance
        self.position        = 0
        self.entry_price     = 0.0
        self.entry_step      = SEQUENCE_LENGTH
        self.prev_unrealized = 0.0
        self.trades          = []
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        """
        Execute one step.

        The agent names a TARGET position, not a command.
          action=0 → target: flat
          action=1 → target: long
          action=2 → target: short

        We compute what transition is needed and execute it.
        """
        price  = float(self.prices[self.current_step])
        reward = 0.0

        # ── desired position from action ──────────────────────────────────────
        target = {0: 0, 1: 1, 2: -1}[action]

        if target == self.position:
            # ── no change — just hold ─────────────────────────────────────────
            reward = self._hold_reward(price)

        elif target == 0:
            # ── close current position ────────────────────────────────────────
            reward = self._close_position(price, reason='signal')

        elif self.position == 0:
            # ── enter new position from flat ──────────────────────────────────
            self._open_position(price, direction=target)
            self.prev_unrealized = 0.0

        else:
            # ── flip position (e.g. long → short) ────────────────────────────
            # Step 1: close the current leg (realised reward)
            reward += self._close_position(price, reason='flip')
            # Step 2: immediately open the other leg
            self._open_position(price, direction=target)
            self.prev_unrealized = 0.0

        # ── stop-loss check ───────────────────────────────────────────────────
        if self.position != 0:
            sl_reward = self._check_stop_loss(price)
            reward += sl_reward

        # ── advance step ──────────────────────────────────────────────────────
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated  = self.balance <= self.initial_balance * 0.5   # −50% ruin guard

        obs  = self._get_obs()
        info = self._get_info()

        if terminated or truncated:
            # If still in a position at end of episode, close at last price
            if self.position != 0:
                close_price = float(self.prices[self.current_step - 1])
                self._close_position(close_price, reason='eod')
            info['final_balance'] = float(self.balance)
            info['total_return']  = float(self.balance / self.initial_balance - 1.0)
            info['n_trades']      = len(self.trades)

        return obs, float(reward), terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────────
    # Internal execution helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _open_position(self, price: float, direction: int):
        """
        Enter a long (direction=1) or short (direction=-1) position.
        Fee is immediately paid on entry; entry_price reflects cost basis.
        """
        # direction=1 → buy at ask ≈ price*(1+fee)
        # direction=-1 → sell at bid ≈ price*(1-fee)
        self.position    = direction
        self.entry_price = price * (1 + direction * self.fee)
        self.entry_step  = self.current_step

    def _close_position(self, price: float, reason: str = 'signal') -> float:
        """
        Close the current open position, update balance, record trade.

        Returns the reward signal for this close event.
        """
        if self.position == 0:
            return 0.0

        # ── realised PnL as a fraction of entry ───────────────────────────────
        if self.position == 1:   # closing long
            pnl_pct = (price - self.entry_price) / self.entry_price - self.fee
        else:                    # closing short
            pnl_pct = (self.entry_price - price) / self.entry_price - self.fee

        # ── update portfolio balance ──────────────────────────────────────────
        old_balance      = self.balance
        self.balance     = self.balance * (1.0 + pnl_pct)
        new_balance      = self.balance

        # ── reward: log-return (research standard for financial RL) ──────────
        # log(new/old) ≈ pnl_pct for small pnl_pct, but log-form is better
        # because it is additive across steps and avoids scale drift.
        log_return = np.log(max(new_balance, 1e-8) / max(old_balance, 1e-8))

        if log_return >= 0:
            reward = self.PROFIT_SCALE * log_return
        else:
            # Loss trades are penalised harder (LOSS_SCALE > PROFIT_SCALE).
            # This is the correct asymmetry: it teaches the agent that
            # losing money is more costly than gaining money is beneficial.
            # Reference: Bandarupalli 2025 (SSRN) uses transaction-cost +
            # volatility-sensitive penalty; we simplify to a fixed multiplier.
            reward = self.LOSS_SCALE * log_return   # log_return is negative here

        # ── record trade ──────────────────────────────────────────────────────
        self.trades.append({
            'direction':   self.position,
            'entry_price': self.entry_price,
            'exit_price':  price,
            'entry_step':  self.entry_step,
            'exit_step':   self.current_step,
            'pnl_pct':     pnl_pct,
            'log_return':  log_return,
            'reason':      reason,
        })

        # ── reset position state ──────────────────────────────────────────────
        self.position        = 0
        self.entry_price     = 0.0
        self.prev_unrealized = 0.0

        return reward

    def _hold_reward(self, price: float) -> float:
        """
        Dense reward while holding or sitting flat.

        While IN a position:
            reward = DENSE_SCALE * Δ(unrealized_pnl)
            This gives the agent step-by-step signal about whether the
            trade is moving for or against it — without waiting until close.
            We scale small (0.05) so realized rewards still dominate.

        While FLAT:
            reward = FLAT_PENALTY (tiny negative)
            This prevents the degenerate "never trade" policy. The magnitude
            is deliberately tiny so it doesn't override the realized signal.
        """
        if self.position == 0:
            return self.FLAT_PENALTY

        # Unrealized PnL as fraction of entry
        if self.position == 1:
            unrealized = (price - self.entry_price) / self.entry_price
        else:
            unrealized = (self.entry_price - price) / self.entry_price

        # Change in unrealized (dense signal)
        delta       = unrealized - self.prev_unrealized
        reward      = self.DENSE_SCALE * delta
        self.prev_unrealized = unrealized
        return reward

    def _check_stop_loss(self, price: float) -> float:
        """
        Force-close position if unrealized loss exceeds STOP_LOSS_PCT.
        Returns the (negative) reward from the forced close.
        """
        if self.position == 0:
            return 0.0

        if self.position == 1:
            pnl = (price - self.entry_price) / self.entry_price
        else:
            pnl = (self.entry_price - price) / self.entry_price

        if pnl < -STOP_LOSS_PCT:
            return self._close_position(price, reason='stop_loss')

        return 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────────────

    def _get_market_state(self) -> np.ndarray:
        """
        Extract the 64-dim (or raw feature) market state.

        If use_lstm=True:  run BiLSTM on the last SEQUENCE_LENGTH candles.
        Otherwise:         return the raw feature vector for this step.
        """
        if self.use_lstm:
            import torch
            start = max(0, self.current_step - SEQUENCE_LENGTH)
            seq   = self.features[start:self.current_step]

            if len(seq) < SEQUENCE_LENGTH:
                pad = np.zeros((SEQUENCE_LENGTH - len(seq), seq.shape[1]), dtype=np.float32)
                seq = np.vstack([pad, seq])

            device = next(self.lstm_model.parameters()).device
            x      = torch.as_tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            state  = self.lstm_model.extract_features(x).squeeze()
            return np.asarray(state, dtype=np.float32)
        else:
            return self.features[self.current_step].copy()

    def _get_obs(self) -> np.ndarray:
        """
        Build the full 70-dim observation vector.

        Market state (64-dim) + account state (6-dim):
          [64] is_flat             — 1 if flat, else 0
          [65] is_long             — 1 if long, else 0
          [66] is_short            — 1 if short, else 0
          [67] unrealized_pnl      — current open trade return, clipped ±0.5
          [68] balance_ratio       — (balance/initial_balance) - 1, clipped ±2
          [69] steps_in_position   — steps held / MAX_HOLD_STEPS, clipped 0..1

        WHY add these 6 features?
        ──────────────────────────
        Without them, the agent cannot distinguish situations like:
          - "price just broke out" when long  vs  "price just broke out" when flat
          - The optimal action is different in each case, but the agent saw
            identical 64-dim BiLSTM vectors for both.
        With position info, the agent can learn rules like:
          - "I'm long + price rising → hold"
          - "I'm flat + price rising → buy"
        """
        market_state = self._get_market_state()

        # ── position one-hot ──────────────────────────────────────────────────
        is_flat  = float(self.position == 0)
        is_long  = float(self.position == 1)
        is_short = float(self.position == -1)

        # ── unrealized PnL (signed) ───────────────────────────────────────────
        if self.position == 0:
            unrealized = 0.0
        else:
            price = float(self.prices[self.current_step])
            if self.position == 1:
                unrealized = (price - self.entry_price) / max(self.entry_price, 1e-8)
            else:
                unrealized = (self.entry_price - price) / max(self.entry_price, 1e-8)
        unrealized = float(np.clip(unrealized, -0.5, 0.5))

        # ── portfolio performance ─────────────────────────────────────────────
        balance_ratio = float(np.clip(
            self.balance / self.initial_balance - 1.0, -2.0, 2.0
        ))

        # ── time in position ──────────────────────────────────────────────────
        steps_held = float(np.clip(
            (self.current_step - self.entry_step) / self.MAX_HOLD_STEPS,
            0.0, 1.0
        ))

        account_state = np.array(
            [is_flat, is_long, is_short, unrealized, balance_ratio, steps_held],
            dtype=np.float32
        )

        return np.concatenate([market_state, account_state], axis=0)

    # ──────────────────────────────────────────────────────────────────────────
    # Info & helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_info(self) -> dict:
        price = float(self.prices[min(self.current_step, self.n_steps - 1)])
        return {
            'step':          self.current_step,
            'balance':       float(self.balance),
            'position':      self.position,
            'entry_price':   float(self.entry_price),
            'current_price': price,
            'n_trades':      len(self.trades),
            'total_return':  float(self.balance / self.initial_balance - 1.0),
        }

    def get_trade_history(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)

    def render(self):
        if self.render_mode == 'human':
            i = self._get_info()
            pos_str = {0: 'FLAT', 1: 'LONG', -1: 'SHORT'}[self.position]
            print(f"Step {i['step']:5d} | Price={i['current_price']:>10.2f} | "
                  f"Pos={pos_str:5s} | Balance={i['balance']:>10.2f} | "
                  f"Return={i['total_return']*100:>+6.2f}%")


def make_env_v2(df: pd.DataFrame, feature_cols: list, **kwargs):
    """Factory for SubprocVecEnv / DummyVecEnv."""
    def _init():
        return CryptoTradingEnvV2(df, feature_cols, **kwargs)
    return _init
