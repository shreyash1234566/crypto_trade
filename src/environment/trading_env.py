"""
Trading Environment - Custom Gymnasium environment for RL training.

Actions: [Buy=0, Hold=1, Sell=2]
State: 64-dim vector from Bi-LSTM
Reward: Asymmetric (+1 profit, -1.5 loss, -0.01 holding loser)
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
    REWARD_PROFIT, REWARD_LOSS, REWARD_HOLD_WINNING, REWARD_HOLD_LOSING,
    REWARD_UNREALIZED_SCALE, REWARD_FLAT_PENALTY,
    STOP_LOSS_PCT, DEVICE_LSTM
)


class CryptoTradingEnv(gym.Env):
    """
    Cryptocurrency trading environment for PPO agent.
    
    Observations:
        - 64-dim state vector from Bi-LSTM (or raw features if no LSTM)
        
    Actions:
        - 0: Buy (go long)
        - 1: Hold (do nothing)
        - 2: Sell (go short / close long)
        
    Rewards:
        - Profit: +1.0
        - Loss: -1.5 (asymmetric to teach caution)
        - Holding losing position: -0.01 per step
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,  # 0.1% Binance fee
        use_lstm_features: bool = False,
        lstm_model=None,
        render_mode=None,
        reward_profit: float = REWARD_PROFIT,
        reward_loss: float = REWARD_LOSS,
        reward_hold_winning: float = REWARD_HOLD_WINNING,
        reward_hold_losing: float = REWARD_HOLD_LOSING,
        reward_unrealized_scale: float = REWARD_UNREALIZED_SCALE,
        flat_penalty: float = REWARD_FLAT_PENALTY
    ):
        """
        Initialize environment.
        
        Args:
            df: DataFrame with features and price data
            feature_cols: List of feature column names
            initial_balance: Starting balance in USDT
            transaction_fee: Transaction fee as decimal
            use_lstm_features: Whether to use Bi-LSTM for state
            lstm_model: Pre-trained Bi-LSTM model (optional)
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.initial_balance = initial_balance
        self.initial_equity = initial_balance
        self.transaction_fee = transaction_fee
        self.use_lstm_features = use_lstm_features
        self.lstm_model = lstm_model
        self.render_mode = render_mode
        self.reward_profit = reward_profit
        self.reward_loss = reward_loss
        self.reward_hold_winning = reward_hold_winning
        self.reward_hold_losing = reward_hold_losing
        self.reward_unrealized_scale = reward_unrealized_scale
        self.flat_penalty = flat_penalty
        if self.use_lstm_features and self.lstm_model is not None:
            # Ensure model resides on configured device (VecEnv may deepcopy)
            try:
                self.lstm_model.to(DEVICE_LSTM)
                self.lstm_model.eval()
            except Exception as exc:
                raise RuntimeError("Failed to move Bi-LSTM to DEVICE_LSTM") from exc
        
        # Prepare data
        self.prices = self.df['close'].values
        self.features = self.df[feature_cols].values.astype(np.float32)
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        self.n_steps = len(self.df)
        
        # Define spaces
        self.action_space = spaces.Discrete(3)  # Buy, Hold, Sell
        
        if use_lstm_features and lstm_model is not None:
            obs_dim = STATE_DIM  # 64
        else:
            obs_dim = len(feature_cols)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # State tracking
        self.current_step = SEQUENCE_LENGTH
        self.balance = initial_balance
        self.initial_equity = initial_balance
        self.position = 0  # 0 = flat, 1 = long, -1 = short
        self.entry_price = 0.0
        self.entry_step = SEQUENCE_LENGTH
        self.trades = []
        self.prev_unrealized_pnl = 0.0  # Track for dense reward
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = SEQUENCE_LENGTH
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.entry_step = SEQUENCE_LENGTH
        self.trades = []
        self.prev_unrealized_pnl = 0.0
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int):
        """
        Execute one step in the environment.
        
        Args:
            action: 0=Buy, 1=Hold, 2=Sell
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        current_price = self.prices[self.current_step]
        reward = 0.0
        
        # Execute action
        if action == 0:  # Buy
            reward = self._execute_buy(current_price)
        elif action == 2:  # Sell
            reward = self._execute_sell(current_price)
        else:  # Hold
            reward = self._calculate_hold_reward(current_price)
        
        # Check stop loss
        if self.position != 0:
            reward += self._check_stop_loss(current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= self.n_steps - 1
        truncated = self.balance <= 0
        
        obs = self._get_observation()
        info = self._get_info()

        if terminated or truncated:
            final_balance = float(self.balance)
            info['final_balance'] = final_balance
            info['pnl'] = final_balance - float(self.initial_balance)
            info['total_return'] = (
                final_balance / float(self.initial_balance) - 1.0
            )
            if truncated and not terminated:
                info.setdefault('TimeLimit.truncated', True)
            info['episode_return'] = info['total_return']

        return obs, reward, terminated, truncated, info
    
    def _execute_buy(self, price: float) -> float:
        """Execute buy action."""
        reward = 0.0
        
        if self.position == 0:
            # Enter long position
            self.position = 1
            self.entry_price = price * (1 + self.transaction_fee)
            self.entry_step = self.current_step
            self.prev_unrealized_pnl = 0.0
            
        elif self.position == -1:
            # Close short position
            pnl_pct = (self.entry_price - price) / self.entry_price
            pnl_pct -= self.transaction_fee  # Fee for closing
            
            # Realized PnL reward (no scaling, direct PnL %)
            if pnl_pct > 0:
                reward = self.reward_profit * pnl_pct
            else:
                reward = self.reward_loss * abs(pnl_pct)
            
            prev_balance = self.balance
            pnl_value = prev_balance * pnl_pct
            self.balance = prev_balance * (1 + pnl_pct)
            self._log_trade(
                trade_type='short_close',
                direction=-1,
                entry_price=self.entry_price,
                exit_price=price,
                pnl_pct=pnl_pct,
                pnl_value=pnl_value,
                reason='close_short'
            )
            
            self.position = 0
            self.entry_price = 0.0
            self.entry_step = self.current_step
            self.prev_unrealized_pnl = 0.0
        
        return reward
    
    def _execute_sell(self, price: float) -> float:
        """Execute sell action."""
        reward = 0.0
        
        if self.position == 0:
            # Enter short position
            self.position = -1
            self.entry_price = price * (1 - self.transaction_fee)
            self.entry_step = self.current_step
            self.prev_unrealized_pnl = 0.0
            
        elif self.position == 1:
            # Close long position
            pnl_pct = (price - self.entry_price) / self.entry_price
            pnl_pct -= self.transaction_fee  # Fee for closing
            
            # Realized PnL reward (no scaling, direct PnL %)
            if pnl_pct > 0:
                reward = self.reward_profit * pnl_pct
            else:
                reward = self.reward_loss * abs(pnl_pct)
            
            prev_balance = self.balance
            pnl_value = prev_balance * pnl_pct
            self.balance = prev_balance * (1 + pnl_pct)
            self._log_trade(
                trade_type='long_close',
                direction=1,
                entry_price=self.entry_price,
                exit_price=price,
                pnl_pct=pnl_pct,
                pnl_value=pnl_value,
                reason='close_long'
            )
            
            self.position = 0
            self.entry_price = 0.0
            self.entry_step = self.current_step
            self.prev_unrealized_pnl = 0.0
        
        return reward
    
    def _calculate_hold_reward(self, price: float) -> float:
        """Calculate reward for holding with dense PnL shaping.

        When flat (no open position), we add a tiny negative reward to
        discourage the degenerate "never trade" policy while keeping the
        magnitude small enough not to dominate realized PnL.
        """
        if self.position == 0:
            # Penalty for staying flat to encourage exploration
            # Keep magnitude small so realized PnL still dominates.
            return self.flat_penalty
        
        # Calculate unrealized PnL for open position
        if self.position == 1:
            unrealized_pnl = (price - self.entry_price) / self.entry_price
        else:
            unrealized_pnl = (self.entry_price - price) / self.entry_price
        
        # Dense reward: change in unrealized PnL (scaled)
        pnl_change = unrealized_pnl - self.prev_unrealized_pnl
        reward = pnl_change * self.reward_unrealized_scale
        
        # Add small incentive/penalty for holding winners/losers
        if unrealized_pnl > 0:
            reward += self.reward_hold_winning
        else:
            reward += self.reward_hold_losing
        
        # Update tracking
        self.prev_unrealized_pnl = unrealized_pnl
        
        return reward
    
    def _check_stop_loss(self, price: float) -> float:
        """Check and execute stop loss if triggered."""
        if self.position == 0:
            return 0.0
        
        if self.position == 1:
            pnl = (price - self.entry_price) / self.entry_price
        else:
            pnl = (self.entry_price - price) / self.entry_price
        
        if pnl < -STOP_LOSS_PCT:
            # Trigger stop loss
            prev_balance = self.balance
            pnl_value = prev_balance * pnl
            direction = self.position
            self.balance = prev_balance * (1 + pnl)
            self._log_trade(
                trade_type='stop_loss',
                direction=direction,
                entry_price=self.entry_price,
                exit_price=price,
                pnl_pct=pnl,
                pnl_value=pnl_value,
                reason='stop_loss'
            )
            self.position = 0
            self.entry_price = 0.0
            self.entry_step = self.current_step
            self.prev_unrealized_pnl = 0.0
            # Standard loss penalty (not extra)
            return self.reward_loss * abs(pnl)
        
        return 0.0

    def _log_trade(
        self,
        *,
        trade_type: str,
        direction: int,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        pnl_value: float,
        reason: str
    ) -> None:
        """Append a rich trade record for downstream analysis."""
        duration = max(1, self.current_step - getattr(self, 'entry_step', self.current_step))
        trade = {
            'type': trade_type,
            'direction': direction,
            'reason': reason,
            'entry': entry_price,
            'exit': exit_price,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_step': getattr(self, 'entry_step', self.current_step),
            'exit_step': self.current_step,
            'duration': duration,
            'pnl_pct': pnl_pct,
            'pnl_value': pnl_value,
            'pnl': pnl_value
        }
        self.trades.append(trade)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.use_lstm_features and self.lstm_model is not None:
            # Get sequence and extract LSTM features
            import torch
            start_idx = max(0, self.current_step - SEQUENCE_LENGTH)
            sequence = self.features[start_idx:self.current_step]
            
            # Pad if necessary
            if len(sequence) < SEQUENCE_LENGTH:
                padding = np.zeros((SEQUENCE_LENGTH - len(sequence), sequence.shape[1]))
                sequence = np.vstack([padding, sequence])
            
            # Extract features on same device as LSTM parameters to avoid mismatches
            model_device = next(self.lstm_model.parameters()).device
            x = torch.as_tensor(sequence, dtype=torch.float32).unsqueeze(0).to(model_device)
            obs = self.lstm_model.extract_features(x).squeeze()
        else:
            # Use raw features from current step
            obs = self.features[self.current_step]
        
        return np.asarray(obs, dtype=np.float32)
    
    def _get_info(self) -> dict:
        """Get info dict."""
        return {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'entry_price': self.entry_price,
            'current_price': self.prices[self.current_step],
            'n_trades': len(self.trades),
            'total_return': (self.balance - self.initial_balance) / self.initial_balance
        }
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
    
    def render(self):
        """Render environment state."""
        if self.render_mode == 'human':
            info = self._get_info()
            print(f"Step {info['step']}: Balance=${info['balance']:.2f}, "
                  f"Position={info['position']}, Price=${info['current_price']:.2f}")


def make_env(df: pd.DataFrame, feature_cols: list, **kwargs):
    """Factory function for creating environments."""
    def _init():
        return CryptoTradingEnv(df, feature_cols, **kwargs)
    return _init


if __name__ == "__main__":
    # Test environment
    import pandas as pd
    
    # Create dummy data
    n_samples = 1000
    df = pd.DataFrame({
        'close': np.random.randn(n_samples).cumsum() + 100,
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
    })
    
    env = CryptoTradingEnv(df, feature_cols=['feature1', 'feature2'])
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Test random actions
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    print(f"\nFinal info: {info}")
    print(f"Trade history:\n{env.get_trade_history()}")
