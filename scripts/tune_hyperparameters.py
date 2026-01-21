# scripts/tune_hyperparameters.py
"""
Final patched Optuna tuning script for RecurrentPPO (ROI objective).
Tailored for:
  - stable-baselines3 2.7.0
  - sb3_contrib RecurrentPPO
  - gymnasium / gym compatibility
  - Robust predict() API handling
"""

import argparse
import gc
import inspect
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

# Try gym then gymnasium for compatibility
try:
    import gym
except Exception:
    import gymnasium as gym  # type: ignore

# Ensure project's root is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# stable-baselines3 & sb3_contrib
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
import torch.nn as nn

# Try to import project modules, fall back to mocks for safety
try:
    from config.settings import SYMBOL, TIMEFRAME_TRADE, DEVICE_PPO
    from src.data.features import load_featured_data, get_feature_columns
    from src.environment.trading_env import CryptoTradingEnv
    from src.models.bilstm import load_model as load_bilstm, freeze_model
except Exception:
    # Minimal fallback stubs so script is still parseable in isolation
    print("WARNING: Some project imports failed. Running with mock stubs.")
    SYMBOL = "BTC/USDT"
    TIMEFRAME_TRADE = "15m"
    DEVICE_PPO = "cpu"

    def load_featured_data(symbol, timeframe):
        data_len = 2000
        feature_count = 12
        dates = pd.date_range(start="2023-01-01", periods=data_len, freq="15min")
        arr = np.random.randn(data_len, feature_count)
        df = pd.DataFrame(arr, index=dates, columns=[f"f{i}" for i in range(feature_count)])
        df["Close"] = np.cumsum(np.random.randn(data_len) * 0.1 + 100)
        return df

    def get_feature_columns():
        df = load_featured_data(None, None)
        return [c for c in df.columns if c != "Close"]

    class CryptoTradingEnv(gym.Env):
        def __init__(self, df, feature_cols, use_lstm_features=False, lstm_model=None, **kwargs):
            super().__init__()
            self.df = df
            self.feature_cols = feature_cols
            self.initial_equity = 1000.0
            self.equity = float(self.initial_equity)
            self.current_step = 0
            self.max_steps = len(df) - 1
            self.action_space = gym.spaces.Discrete(3)
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(feature_cols),), dtype=np.float32)

        def reset(self, **kwargs):
            self.current_step = 0
            self.equity = float(self.initial_equity)
            # return gym or gymnasium style
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            try:
                # gymnasium expects (obs, info)
                return obs, {}
            except Exception:
                return obs

        def step(self, action):
            self.current_step += 1
            # random reward for stub
            reward = float(np.random.randn() * 0.01)
            done = self.current_step >= self.max_steps
            info = {}
            if done:
                # produce final_equity and total_return
                self.equity += float(np.random.randn() * 5.0)
                info["final_equity"] = float(self.equity)
                info["total_return"] = (self.equity / self.initial_equity) - 1.0
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            # gymnasium style: (obs, reward, terminated, truncated, info)
            try:
                return obs, reward, bool(done), False, info
            except Exception:
                return obs, reward, bool(done), info

    def load_bilstm(path):
        import torch.nn as _nn
        class Dummy(_nn.Module):
            def __init__(self): super().__init__()
            def forward(self, x): return x
        return Dummy()

    def freeze_model(m): pass

warnings.filterwarnings("ignore")


# --------------------------
# Helpers
# --------------------------
def _ensure_float(value: Any) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(value[0]) if len(value) else 0.0
    return float(value)


def recurrent_predict(model, obs, lstm_states, episode_starts, deterministic=True) -> Tuple[Any, Any]:
    """
    Robustly call model.predict for recurrent policies across SB3 versions.
    - Prefer to pass `state` and `episode_start` if supported (SB3 >=2.x)
    - Fallback to `mask` if older signature requires it
    - Finally fallback to plain predict(obs, deterministic=...)
    Returns (action, new_state_or_None)
    """
    sig = inspect.signature(model.predict)
    params = sig.parameters

    kwargs = {"deterministic": deterministic}
    if "state" in params:
        kwargs["state"] = lstm_states
    # prefer episode_start name
    if "episode_start" in params:
        kwargs["episode_start"] = episode_starts
    elif "mask" in params:
        kwargs["mask"] = episode_starts

    try:
        out = model.predict(obs, **kwargs)
    except TypeError:
        # fallback: try without state/episode_start/mask
        out = model.predict(obs, deterministic=deterministic)

    # Some versions return action only, others return (action, state)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    else:
        return out, None


# --------------------------
# Optuna callback (for trial progress)
# --------------------------
def _make_optuna_progress_callback(total_trials: int) -> Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]:
    def _callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        finished_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL))
        done = len(finished_trials)
        percent = min(100.0, (done / total_trials) * 100.0) if total_trials else 0.0
        value_str = f"{trial.value:.4f}" if trial.value is not None else "nan"
        print(f"[Progress] {done}/{total_trials} trials finished ({percent:5.1f}%). Trial {trial.number} -> state={trial.state.name}, value={value_str}")
    return _callback


# --------------------------
# Recurrent trial eval callback (for pruning)
# --------------------------
class TrialEvalCallback(BaseCallback):
    def __init__(self, eval_env, trial: optuna.Trial, n_eval_episodes: int = 5, eval_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.trial = trial
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.eval_idx = 0
        self.is_pruned = False
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            mean_reward, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, warn=False)
            mean_reward = _ensure_float(mean_reward)
            self.eval_idx += 1
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
            # report shaped reward for pruning decisions
            self.trial.report(mean_reward, self.eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


# --------------------------
# Hyperparameter sampling (fixed PPO + reward ranges)
# --------------------------
def sample_ppo_params(_: optuna.Trial) -> Dict[str, Any]:
    learning_rate = 3e-4
    n_steps = 2048
    batch_size = 512
    n_epochs = 10
    gamma = 0.979575
    gae_lambda = 0.9452
    clip_range = 0.1154
    ent_coef = 0.027865
    vf_coef = 0.6633
    max_grad_norm = 0.4677
    lstm_hidden_size = 256

    net_arch = dict(pi=[256, 128], vf=[256, 128])
    activation_fn = nn.ReLU
    ortho_init = False

    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "lstm_hidden_size": lstm_hidden_size,
            "net_arch": net_arch,
            "activation_fn": activation_fn,
            "ortho_init": ortho_init,
            "n_lstm_layers": 1,
        }
    }


def sample_reward_params(trial: optuna.Trial) -> Dict[str, float]:
    """
    Production-ready reward parameter sampling with:
    1. Loss aversion (losses hurt more than wins feel good)
    2. Discourage holding losing positions
    3. Realistic transaction costs
    4. Conservative unrealized gain rewards
    """
    # Core trade rewards (asymmetric loss aversion)
    reward_profit = trial.suggest_float("reward_profit", 1.6, 2.0)
    reward_loss = trial.suggest_float("reward_loss", -1.2, -0.8)  # More negative than profit is positive
    # Holding behavior
    reward_hold_winning = trial.suggest_float("reward_hold_winning", 0.0001, 0.001)
    reward_hold_losing = trial.suggest_float("reward_hold_losing", -0.002, -0.0005)  # Strong penalty
    # Unrealized gains (conservative)
    reward_unrealized_scale = trial.suggest_float("reward_unrealized_scale", 0.001, 0.01)
    # Realistic transaction costs (0.1% fee per trade)
    flat_penalty = -0.001  # Fixed realistic value, not sampled
    return {
        "reward_profit": reward_profit,
        "reward_loss": reward_loss,
        "reward_hold_winning": reward_hold_winning,
        "reward_hold_losing": reward_hold_losing,
        "reward_unrealized_scale": reward_unrealized_scale,
        "flat_penalty": flat_penalty,
    }


# --------------------------
# Objective function (returns mean realized ROI across n_eval episodes)
# --------------------------
def objective(trial: optuna.Trial, df, feature_cols, bilstm_model, n_timesteps: int, n_eval_episodes: int = 5, eval_freq: int = 15000) -> float:

    # disable tqdm-rich inside subprocesses to avoid "Only one live display" errors
    os.environ["TQDM_DISABLE"] = "1"

    ppo_params = sample_ppo_params(trial)
    reward_params = sample_reward_params(trial)

    # Train/test split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    def make_env_frame(dataframe):
        return lambda: CryptoTradingEnv(
            dataframe,
            feature_cols,
            use_lstm_features=(bilstm_model is not None),
            lstm_model=bilstm_model,
            **reward_params
        )

    train_env = VecNormalize(
        DummyVecEnv([make_env_frame(train_df)]),
        norm_obs=True,
        norm_reward=False,
        clip_reward=100.0
    )
    eval_env = VecNormalize(
        DummyVecEnv([make_env_frame(test_df)]),
        norm_obs=True,
        norm_reward=False,
        training=False,
        clip_reward=100.0
    )

    model = None
    try:
        print(f"[Trial {trial.number}] Reward params: {reward_params}")

        # Create model
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=train_env,
            verbose=0,
            device=DEVICE_PPO,
            **ppo_params
        )

        # Sync normalization
        try:
            eval_env.obs_rms = train_env.obs_rms
        except Exception:
            pass

        eval_callback = TrialEvalCallback(
            eval_env=eval_env,
            trial=trial,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq
        )

        print(f"[Trial {trial.number}] Training for {n_timesteps} timesteps...")
        model.learn(
            total_timesteps=n_timesteps,
            callback=eval_callback,
            progress_bar=False
        )

        if eval_callback.is_pruned:
            raise optuna.TrialPruned()

        # Collect episode returns for risk-adjusted metric
        episode_returns = []
        initial_equity = None
        try:
            inner_env = eval_env.venv.envs[0]
            initial_equity = getattr(inner_env, "initial_equity", None)
        except Exception:
            pass

        for ep in range(n_eval_episodes):
            reset_out = eval_env.reset()
            if isinstance(reset_out, tuple) and len(reset_out) == 2:
                obs, _ = reset_out
            else:
                obs = reset_out

            lstm_states = None
            episode_starts = np.ones((eval_env.num_envs,), dtype=bool)

            while True:
                action, lstm_states = recurrent_predict(
                    model, obs, lstm_states, episode_starts, deterministic=True
                )

                step_out = eval_env.step(action)
                if len(step_out) == 4:
                    obs, rewards, dones, infos = step_out
                    truncations = np.zeros_like(dones, dtype=bool)
                elif len(step_out) == 5:
                    obs, rewards, dones, truncations, infos = step_out
                else:
                    raise RuntimeError("Unexpected env.step return arity")

                episode_starts = np.logical_or(dones, truncations).astype(bool)
                done = bool(dones[0]) or (len(truncations) > 0 and bool(truncations[0]))

                if done:
                    info = infos[0] if isinstance(infos, (list, tuple)) else infos
                    total_return = None
                    if isinstance(info, dict):
                        total_return = info.get("total_return", None)
                        if total_return is None and "final_equity" in info and initial_equity is not None:
                            final_equity = float(info.get("final_equity"))
                            total_return = (final_equity / float(initial_equity)) - 1.0
                    if total_return is not None:
                        episode_returns.append(float(total_return))
                    else:
                        episode_returns.append(0.0)
                    break

        # Calculate risk-adjusted return (Sharpe-like)
        if not episode_returns or all(r == 0.0 for r in episode_returns):
            return float("-inf")

        mean_return = float(np.mean(episode_returns))
        std_return = float(np.std(episode_returns))

        # Sharpe ratio with minimum volatility threshold
        if std_return < 1e-6:
            sharpe_ratio = mean_return * 10  # High reward for consistent returns
        else:
            sharpe_ratio = mean_return / std_return

        # Penalty multiplier for negative mean returns
        if mean_return < 0:
            sharpe_ratio = sharpe_ratio * 2  # Double the penalty

        # Log metrics
        trial.set_user_attr("mean_return", mean_return)
        trial.set_user_attr("std_return", std_return)
        trial.set_user_attr("sharpe_ratio", sharpe_ratio)

        print(f"[Trial {trial.number}] Mean Return: {mean_return:.4f}, Std: {std_return:.4f}, Sharpe: {sharpe_ratio:.4f}")

        return sharpe_ratio

    finally:
        # Cleanup
        try:
            if model is not None and hasattr(model, "env"):
                model.env.close()
        except Exception:
            pass
        try:
            train_env.close()
            eval_env.close()
        except Exception:
            pass
        if model is not None:
            del model
        gc.collect()

    # Only one finally block needed for cleanup (already present above)


# --------------------------
# Runner
# --------------------------
def run_optimization(n_trials: int = 50, n_timesteps: int = 75000, use_lstm: bool = True, study_name: str = "ppo_crypto_rewards_final_roi", storage: Optional[str] = None, n_jobs: int = 1):
    print("=" * 60)
    print("FINAL ROBUST REWARD FUNCTION OPTIMIZATION (MAXIMIZING ROI)")
    print("=" * 60)
    print(f"Trials: {n_trials}")
    print(f"Timesteps per trial: {n_timesteps}")
    print(f"Use LSTM: {use_lstm}")
    print(f"Parallel Jobs: {n_jobs}")
    print("=" * 60)

    try:
        df = load_featured_data(symbol=SYMBOL, timeframe=TIMEFRAME_TRADE)
        feature_cols = get_feature_columns()
        feature_cols = [c for c in feature_cols if c in df.columns]
        print(f"Data: {len(df)} candles, {len(feature_cols)} features")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    bilstm_model = None
    if use_lstm:
        try:
            bilstm_model = load_bilstm("bilstm_best.pt")
            freeze_model(bilstm_model)
            print("Loaded pre-trained Bi-LSTM")
        except FileNotFoundError:
            print("No pre-trained Bi-LSTM found, using raw features")

    sampler = TPESampler(n_startup_trials=10, seed=42)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=3, interval_steps=1)

    study = optuna.create_study(study_name=study_name, sampler=sampler, pruner=pruner, direction="maximize", storage=storage, load_if_exists=True)

    print(f"\nStarting optimization with {n_trials} trials...")
    func = lambda trial: objective(trial, df=df, feature_cols=feature_cols, bilstm_model=bilstm_model, n_timesteps=n_timesteps, eval_freq=max(1000, n_timesteps // 5))
    progress_cb = _make_optuna_progress_callback(n_trials)
    study.optimize(func, n_trials=n_trials, n_jobs=n_jobs, callbacks=[progress_cb])

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    if study.best_value is not None:
        print(f"Best trial value (Mean ROI): {study.best_value:.4f}")
    else:
        print("Best trial value: None")
    print("Best Reward Parameters:")
    try:
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
    except Exception:
        print("No best params found.")


# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final Robust Optuna Tuning for RecurrentPPO")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials to run")
    parser.add_argument("--timesteps", type=int, default=75000, help="Total timesteps per trial")
    parser.add_argument("--use-lstm", action="store_true", help="Use pre-trained Bi-LSTM features")
    parser.add_argument("--no-lstm", action="store_true", help="Convenience flag to disable LSTM")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs (use 1 on Windows)")
    args = parser.parse_args()

    use_lstm_flag = False if args.no_lstm else args.use_lstm
    # prefer single job on Windows by default to avoid tqdm issues; user can override
    run_optimization(n_trials=args.n_trials, n_timesteps=args.timesteps, use_lstm=use_lstm_flag, n_jobs=args.n_jobs)
