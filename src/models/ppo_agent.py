"""
PPO Agent - Proximal Policy Optimization with custom Bi-LSTM feature extractor.

Uses stable-baselines3 PPO with a custom policy network that incorporates
the pre-trained Bi-LSTM as a frozen feature extractor.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Type
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from config.settings import (
    DEVICE_PPO, STATE_DIM, MODELS_DIR,
    PPO_LEARNING_RATE, PPO_N_STEPS, PPO_BATCH_SIZE,
    PPO_N_EPOCHS, PPO_GAMMA, PPO_GAE_LAMBDA, PPO_ENT_COEF,
    PPO_CLIP_RANGE, PPO_VF_COEF, PPO_MAX_GRAD_NORM,
    PPO_LSTM_HIDDEN_SIZE, PPO_N_LSTM_LAYERS,
    PPO_ENT_COEF_INITIAL, PPO_ENT_COEF_FINAL, PPO_USE_ENT_SCHEDULE,
    PPO_NET_ARCH, PPO_ACTIVATION_FN, PPO_ORTHO_INIT
)
from src.models.bilstm import BiLSTMFeatureExtractor, load_model as load_bilstm


class BiLSTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor that uses pre-trained Bi-LSTM.
    
    If no LSTM model is provided, uses a simple MLP.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = STATE_DIM,
        bilstm_model: BiLSTMFeatureExtractor = None
    ):
        super().__init__(observation_space, features_dim)
        
        self.bilstm_model = bilstm_model
        input_dim = observation_space.shape[0]
        
        if bilstm_model is not None:
            # Use pre-trained Bi-LSTM (frozen)
            self.extractor = bilstm_model
            # Freeze LSTM weights
            for param in self.extractor.parameters():
                param.requires_grad = False
        else:
            # Fallback: simple MLP
            self.extractor = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, features_dim),
                nn.Tanh()
            )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.bilstm_model is not None:
            # LSTM expects (batch, seq, features) but we have (batch, features)
            # The observation should already be the LSTM output from the env
            return observations
        else:
            return self.extractor(observations)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to Tensorboard.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        # Log additional info from environment
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            if 'total_return' in ep_info:
                self.logger.record('trade/total_return', ep_info['total_return'])
            if 'n_trades' in ep_info:
                self.logger.record('trade/n_trades', ep_info['n_trades'])
        return True


def _get_vecnorm_path(model_name: str) -> Path:
    """Construct VecNormalize stats filepath for a given model name."""
    stem = Path(model_name).stem
    if stem.startswith('ppo_fold_'):
        fold_suffix = stem.replace('ppo_', '', 1)
        return MODELS_DIR / f"vecnorm_{fold_suffix}.pkl"
    return MODELS_DIR / f"vecnorm_{stem}.pkl"


def _save_vecnormalize_stats(model: RecurrentPPO, model_name: str):
    """Persist VecNormalize statistics alongside model weights."""
    vec_env = model.get_env()
    if not isinstance(vec_env, VecNormalize):
        print("VecNormalize stats not saved: training env is not VecNormalize.")
        return
    vec_path = _get_vecnorm_path(model_name)
    vec_env.save(str(vec_path))
    print(f"VecNormalize stats saved to {vec_path}")

def linear_schedule(initial_value: float, final_value: float = 0.0):
    """
    Linear learning rate / entropy schedule.
    
    Args:
        initial_value: Initial value
        final_value: Final value at the end of training
        
    Returns:
        Schedule function that takes progress_remaining (1.0 -> 0.0)
    """
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return func


def create_ppo_agent(
    env: gym.Env,
    bilstm_model: BiLSTMFeatureExtractor = None,
    tensorboard_log: str = None,
    total_timesteps: int = 500000
) -> RecurrentPPO:
    """
    Create RecurrentPPO agent with custom feature extractor.
    
    Uses research-optimized hyperparameters from config/settings.py:
    - Higher learning rate (3e-4) with linear decay
    - Entropy coefficient schedule to prevent premature convergence
    - Smaller batch size for LSTM policies
    - Fewer epochs to prevent overfitting
    
    Args:
        env: Gymnasium environment
        bilstm_model: Pre-trained Bi-LSTM model (optional)
        tensorboard_log: Path for tensorboard logs
        total_timesteps: Total training timesteps (for schedule calculation)
        
    Returns:
        RecurrentPPO agent
    """
    # Wrap environment
    if not isinstance(env, DummyVecEnv):
        env = DummyVecEnv([lambda: env])
    
    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Parse network architecture
    if PPO_NET_ARCH == "small":
        net_arch = dict(pi=[64, 64], vf=[64, 64])
    elif PPO_NET_ARCH == "medium":
        net_arch = dict(pi=[128, 128], vf=[128, 128])
    elif PPO_NET_ARCH == "large":
        net_arch = dict(pi=[256, 128], vf=[256, 128])
    else:
        # Default fallback
        net_arch = dict(pi=[128, 128], vf=[128, 128])
        
    # Parse activation function
    activation_fn = nn.Tanh if PPO_ACTIVATION_FN == "tanh" else nn.ReLU

    # Policy kwargs with optimized architecture
    policy_kwargs = dict(
        features_extractor_class=BiLSTMFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=STATE_DIM,
            bilstm_model=bilstm_model
        ),
        net_arch=net_arch,
        activation_fn=activation_fn,
        lstm_hidden_size=PPO_LSTM_HIDDEN_SIZE,
        n_lstm_layers=PPO_N_LSTM_LAYERS,
        ortho_init=PPO_ORTHO_INIT,
    )
    
    # Learning rate schedule (linear decay)
    lr_schedule = linear_schedule(PPO_LEARNING_RATE, final_value=1e-5)
    
    # Note: RecurrentPPO does not support callable schedules for ent_coef
    # (only learning_rate supports schedules), so use constant value
    ent_coef = PPO_ENT_COEF
    
    # Create RecurrentPPO agent with optimized hyperparameters
    model = RecurrentPPO(
        policy='MlpLstmPolicy',
        env=env,
        learning_rate=lr_schedule,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        clip_range=PPO_CLIP_RANGE,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=ent_coef,
        vf_coef=PPO_VF_COEF,
        max_grad_norm=PPO_MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log or str(MODELS_DIR / 'tensorboard'),
        verbose=1,
        device=DEVICE_PPO
    )
    
    return model


def train_ppo(
    model: RecurrentPPO,
    total_timesteps: int = 100000,
    eval_env: gym.Env = None,
    eval_freq: int = 10000,
    save_path: str = None
) -> RecurrentPPO:
    """
    Train RecurrentPPO agent.
    
    Args:
        model: RecurrentPPO model
        total_timesteps: Total training timesteps
        eval_env: Evaluation environment
        eval_freq: Evaluation frequency
        save_path: Path to save best model
        
    Returns:
        Trained model
    """
    callbacks = [TensorboardCallback()]
    
    # Add evaluation callback if eval env provided
    if eval_env is not None:
        if not isinstance(eval_env, VecEnv):
            # Wrap with Monitor if not already wrapped (for correct logging)
            if not isinstance(eval_env, Monitor):
                eval_env = Monitor(eval_env)
            eval_env = DummyVecEnv([lambda: eval_env])
        
        # Wrap eval env with VecNormalize to match training env
        # Use same normalization stats as training but don't update them
        if not isinstance(eval_env, VecNormalize):
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        
        # Sync normalization stats from training env
        train_env = model.get_env()
        if isinstance(train_env, VecNormalize):
            eval_env.obs_rms = train_env.obs_rms
            # We don't sync reward normalization for evaluation as we want raw rewards or 
            # at least consistent with training but usually we care about raw performance.
            # However, the agent expects normalized inputs.
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path or str(MODELS_DIR),
            log_path=str(MODELS_DIR / 'eval_logs'),
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )

    # Persist VecNormalize stats for the latest/best model if saved via callback
    best_dir = Path(save_path) if save_path is not None else MODELS_DIR
    best_path = best_dir / 'best_model'
    if best_path.with_suffix('.zip').exists():
        _save_vecnormalize_stats(model, best_path.name)
    
    return model


def save_ppo_agent(model: RecurrentPPO, filename: str = 'ppo_agent'):
    """Save RecurrentPPO agent."""
    filepath = MODELS_DIR / filename
    model.save(str(filepath))
    print(f"RecurrentPPO agent saved to {filepath}")
    _save_vecnormalize_stats(model, filename)


def load_ppo_agent(
    env: gym.Env,
    filename: str = 'ppo_agent',
    bilstm_model: BiLSTMFeatureExtractor = None
) -> Tuple[RecurrentPPO, VecNormalize]:
    """Load RecurrentPPO agent along with its VecNormalize statistics."""
    filepath = MODELS_DIR / filename
    vecnorm_path = _get_vecnorm_path(filename)
    
    # Wrap env for vectorized inference
    if not isinstance(env, DummyVecEnv):
        env = DummyVecEnv([lambda: env])
    
    if vecnorm_path.exists():
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training = False
        env.norm_reward = False
    else:
        print(f"Warning: VecNormalize stats not found for {filename}. Using fresh statistics.")
        env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
    
    # Custom objects for loading
    custom_objects = {
        'policy_kwargs': dict(
            features_extractor_class=BiLSTMFeaturesExtractor,
            features_extractor_kwargs=dict(
                features_dim=STATE_DIM,
                bilstm_model=bilstm_model
            )
        )
    }
    
    model = RecurrentPPO.load(str(filepath), env=env, custom_objects=custom_objects, device=DEVICE_PPO)
    print(f"RecurrentPPO agent loaded from {filepath}")
    return model, env


def evaluate_agent(
    model: RecurrentPPO,
    env: gym.Env,
    n_episodes: int = 10
) -> Dict[str, float]:
    """
    Evaluate agent performance.
    
    Args:
        model: Trained PPO model
        env: Evaluation environment
        n_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary of metrics
    """
    all_rewards = []
    all_returns = []
    all_trades = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        lstm_states = None
        # For VecEnv, num_envs is available
        num_envs = env.num_envs
        episode_starts = np.ones((num_envs,), dtype=bool)
        
        while not done:
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts,
                deterministic=True
            )
            # VecEnv step returns obs, rewards, dones, infos
            obs, rewards, dones, infos = env.step(action)
            
            # We assume single environment for evaluation
            episode_reward += rewards[0]
            done = dones[0]
            episode_starts = dones
            
            # Get info from the first env
            info = infos[0]
        
        all_rewards.append(episode_reward)
        all_returns.append(info.get('total_return', 0))
        all_trades.append(info.get('n_trades', 0))
    
    metrics = {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_return': np.mean(all_returns),
        'std_return': np.std(all_returns),
        'mean_trades': np.mean(all_trades),
        'win_rate': sum(1 for r in all_returns if r > 0) / len(all_returns)
    }
    
    print("\nEvaluation Results:")
    print(f"  Mean Reward: {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
    print(f"  Mean Return: {metrics['mean_return']*100:.2f}% +/- {metrics['std_return']*100:.2f}%")
    print(f"  Mean Trades: {metrics['mean_trades']:.1f}")
    print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
    
    return metrics


if __name__ == "__main__":
    # Test agent creation
    from src.environment.trading_env import CryptoTradingEnv
    
    # Create dummy environment
    n_samples = 1000
    df = pd.DataFrame({
        'close': np.random.randn(n_samples).cumsum() + 100,
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
    })
    
    env = CryptoTradingEnv(df, feature_cols=['feature1', 'feature2'])
    
    # Create agent
    model = create_ppo_agent(env)
    print(f"Model created: {model}")
    
    # Test training for a few steps
    model.learn(total_timesteps=1000, progress_bar=True)
    print("Training test passed!")
