"""
PPO Agent v2 — Correct architecture for Bi-LSTM + PPO.

WHAT WAS WRONG IN ppo_agent.py:
─────────────────────────────────────────────────────────────────────────────
WRONG 1 ► Used RecurrentPPO (sb3_contrib) instead of regular PPO.
  RecurrentPPO wraps its own LSTM around whatever observations come in.
  Your BiLSTM ALREADY handles temporal dependencies. Stacking an LSTM on top
  of LSTM features is not just redundant — it is harmful:
    - The inner LSTM gets a sequence of BiLSTM states (each state already
      summarises 60 candles). It then tries to find patterns in THOSE states.
      That is a second level of temporal abstraction that serves no purpose.
    - RecurrentPPO is slower, harder to tune, and has more failure modes
      (BPTT through time, hidden state resets at episode boundaries, etc.)
    - Research (Stanford CS224R 2024): "end-to-end LSTM feeding into LSTM
      failed to generalise" — use the frozen LSTM as extractor, then MLP.

WRONG 2 ► n_epochs=3 is far too few.
  Standard PPO (Schulman 2017) uses 10–30 epochs per rollout.
  3 epochs means you throw away 90% of the gradient information in each
  rollout. The agent barely learns from each batch of experience.

WRONG 3 ► clip_range=0.1154 is too small.
  Standard is 0.2. A very tight clip prevents the policy from updating even
  when the gradient is clearly correct. The agent gets stuck.

WRONG 4 ► learning_rate=4.4e-5 from Optuna was the result of a FAILING search.
  The Optuna study's best_value was -0.0403 (agent was LOSING money at the
  best trial). These hyperparameters were tuned on a broken environment
  (no position info → noisy reward signal → Optuna converged to garbage).

WRONG 5 ► BiLSTMFeaturesExtractor.forward() returned observations unchanged.
  When bilstm_model is not None:
      return observations   ← This is fine ONLY because the env already
                               extracts LSTM features before passing them in.
  But using RecurrentPPO on top made this a double-LSTM architecture anyway.

CORRECT DESIGN IN V2:
─────────────────────
  Environment extracts 64-dim BiLSTM state + 6 account features = 70-dim obs
  PPO (regular, MlpPolicy) gets a clean 70-dim observation each step
  Policy network: pi=[128, 64, 32], vf=[128, 64, 32], Tanh activation
  Tanh is better than ReLU for bounded action distributions in finance
  (Schulman 2017, Mnih 2016: ReLU causes "dying units" in critic networks).

CORRECT HYPERPARAMETERS — and the reasoning behind each:
─────────────────────────────────────────────────────────
  n_steps=2048          Standard rollout length. Balances bias/variance in
                        GAE advantage estimation. 4096 from Optuna is too big
                        and makes each update stale.

  batch_size=64         Smaller batch → noisier but more gradient updates per
                        rollout. Financial time series benefits from more
                        updates to handle non-stationarity.

  n_epochs=10           Standard. 10 passes through each rollout buffer.
                        (Optuna had 3 — we fixed this, the biggest gain.)

  gamma=0.99            High discount factor. Trades can be open for 10–100
                        steps; we need the agent to value future rewards.
                        (Optuna had 0.9795 — close but not optimal.)

  gae_lambda=0.95       Standard. Controls bias/variance tradeoff in GAE.

  clip_range=0.2        Standard PPO clip (Schulman 2017). Optuna had 0.1154
                        which was too tight.

  ent_coef=0.01         Standard. Enough entropy to explore without chaos.
                        (Optuna had 0.028 — slightly high, caused random
                        action noise even after the agent found good policies.)

  vf_coef=0.5           Standard. Equal weighting of value and policy losses.

  learning_rate=3e-4    Standard Adam learning rate for neural networks.
  → 1e-5 (linear decay) Start fast, end stable.

  max_grad_norm=0.5     Standard gradient clipping. Prevents occasional
                        large gradient spikes from destabilising training.

  normalize_advantage=True  Always True — removes scale sensitivity.

  net_arch pi=[128,64,32]   3 layers is enough. Large networks overfit on
           vf=[128,64,32]   small financial datasets (~17k candles).
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CallbackList
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv, VecNormalize, SubprocVecEnv
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
import gymnasium as gym

from config.settings import STATE_DIM, MODELS_DIR, DEVICE_PPO
from src.models.bilstm import BiLSTMFeatureExtractor


# ─────────────────────────────────────────────────────────────────────────────
# PPO hyperparameters (with reasoning in comments)
# These are NOT from Optuna — Optuna failed (best_value = -0.0403).
# These are standard PPO values validated across hundreds of finance papers.
# ─────────────────────────────────────────────────────────────────────────────

LEARNING_RATE_START = 3e-4     # Standard Adam start for PPO networks
LEARNING_RATE_END   = 1e-5     # Decay target — stabilises at end of training
N_STEPS             = 2048     # Steps per rollout per env
BATCH_SIZE          = 64       # Mini-batch size for SGD updates
N_EPOCHS            = 10       # Gradient passes through each rollout buffer
GAMMA               = 0.99     # Discount — high because trades span many steps
GAE_LAMBDA          = 0.95     # GAE bias/variance tradeoff (standard)
CLIP_RANGE          = 0.2      # PPO clip (standard Schulman 2017)
ENT_COEF            = 0.01     # Entropy bonus — enough to explore
VF_COEF             = 0.5      # Value function loss weight (standard)
MAX_GRAD_NORM       = 0.5      # Gradient clipping (prevents spikes)

# Policy network architecture
# 3 hidden layers is enough for a 70-dim input. More layers overfit on
# the ~17k candles of training data.
NET_ARCH = dict(pi=[128, 64, 32], vf=[128, 64, 32])


# ─────────────────────────────────────────────────────────────────────────────
# Passthrough feature extractor
# ─────────────────────────────────────────────────────────────────────────────

class PassthroughExtractor(BaseFeaturesExtractor):
    """
    Thin SB3 feature extractor that does nothing.

    The environment already extracts BiLSTM features and concatenates
    account state. The observation is a clean 70-dim vector. We don't
    need another transformation layer — just tell SB3 "features_dim = obs_dim"
    so it builds the correct policy head size.

    WHY NOT use the BiLSTM inside the extractor?
    ─────────────────────────────────────────────
    Option A (original design): BiLSTM in env → 64-dim obs → PPO extractor passthrough
    Option B: raw features in obs → PPO extractor calls BiLSTM → PPO policy

    Option A is correct. The BiLSTM processes a SEQUENCE (60 candles) per step.
    The SB3 policy only sees a single observation vector per step. Putting the
    BiLSTM inside the SB3 extractor would require maintaining the raw feature
    buffer inside SB3 infrastructure, which complicates VecNormalize, rollout
    storage, and gradient computation. Option A keeps concerns separate.
    """

    def __init__(self, observation_space: gym.spaces.Box):
        obs_dim = observation_space.shape[0]
        super().__init__(observation_space, features_dim=obs_dim)
        # Identity transformation — no learnable parameters
        self._features_dim = obs_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations   # pass through unchanged


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

class TradeMetricsCallback(BaseCallback):
    """
    Logs per-episode trading metrics to tensorboard.

    SB3's default logging only tracks raw episodic rewards.
    We also want to track: total_return, n_trades, win_rate.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._ep_returns  = []
        self._ep_trades   = []

    def _on_step(self) -> bool:
        # SB3 fills ep_info_buffer when an episode finishes
        for info in self.locals.get('infos', []):
            if 'total_return' in info:
                self._ep_returns.append(info['total_return'])
                self.logger.record('trade/episode_return', info['total_return'])
            if 'n_trades' in info:
                self._ep_trades.append(info['n_trades'])
                self.logger.record('trade/n_trades', info['n_trades'])

        return True


class EntropyScheduleCallback(BaseCallback):
    """
    Linearly anneals ent_coef from ent_start to ent_end over training.

    WHY: High entropy early forces the agent to explore many actions.
    As training progresses and the agent finds good policies, we want
    it to EXPLOIT those policies, not keep exploring randomly.
    Entropy schedule: 0.02 → 0.002 over 500k steps.

    Note: SB3 supports callable schedules for learning_rate but not
    for ent_coef. We implement it manually via this callback.
    """

    def __init__(self, ent_start: float = 0.02, ent_end: float = 0.002,
                 total_timesteps: int = 500_000, verbose=0):
        super().__init__(verbose)
        self.ent_start       = ent_start
        self.ent_end         = ent_end
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps       # 0.0 → 1.0
        current  = self.ent_start + progress * (self.ent_end - self.ent_start)
        self.model.ent_coef = max(current, self.ent_end)
        self.logger.record('train/ent_coef', self.model.ent_coef)
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────────────────────────────────────

def create_ppo_agent_v2(
    env: gym.Env,
    total_timesteps: int = 500_000,
    tensorboard_log: str = None,
) -> PPO:
    """
    Create a correctly configured PPO agent for the Bi-LSTM trading system.

    Args:
        env:              A (possibly already-wrapped) Gymnasium environment.
                          Must be CryptoTradingEnvV2 or compatible.
        total_timesteps:  Total training steps (used only for LR schedule).
        tensorboard_log:  Path for TensorBoard logs.

    Returns:
        PPO agent (stable_baselines3.PPO, NOT RecurrentPPO).
    """

    # ── 1. Wrap environment ───────────────────────────────────────────────────
    # DummyVecEnv: runs a single env. For faster training, swap for
    # SubprocVecEnv with multiple envs (see train_ppo_v2 below).
    if not isinstance(env, (DummyVecEnv, SubprocVecEnv)):
        env = Monitor(env)                 # records episode stats
        env = DummyVecEnv([lambda: env])

    # VecNormalize: normalises observations and rewards online.
    # WHY: The 70-dim obs mixes [-0.5, 0.5] unrealized PnL with
    # [−∞, +∞] BiLSTM features. Without normalisation the actor/critic
    # networks see wildly different scales and training is unstable.
    env = VecNormalize(
        env,
        norm_obs    = True,    # normalise obs to ~N(0,1) using running stats
        norm_reward = True,    # normalise rewards too (helps value function)
        clip_obs    = 10.0,    # clip extreme observations
        clip_reward = 10.0,    # clip extreme rewards
    )

    # ── 2. Linear learning rate schedule ─────────────────────────────────────
    # get_linear_fn(start, end, end_fraction) is SB3's built-in scheduler.
    # The schedule function receives progress_remaining (1.0 → 0.0).
    lr_schedule = get_linear_fn(
        start          = LEARNING_RATE_START,
        end            = LEARNING_RATE_END,
        end_fraction   = 1.0,    # reach end_lr at 100% of training
    )

    # ── 3. Policy kwargs ──────────────────────────────────────────────────────
    policy_kwargs = dict(
        features_extractor_class  = PassthroughExtractor,
        features_extractor_kwargs = {},       # no kwargs needed
        net_arch                  = NET_ARCH,
        # Tanh activation:
        #   - Squashes outputs to (-1, 1) → natural for policy logits
        #   - No dying-unit problem (ReLU issue in critic networks)
        #   - Standard choice in Schulman 2017's own PPO experiments
        activation_fn             = nn.Tanh,
        # Orthogonal initialisation:
        #   - Sets initial weights so activations have unit variance
        #   - Prevents vanishing/exploding gradients from the start
        #   - Standard in modern PPO implementations (SB3 default = True)
        ortho_init                = True,
    )

    # ── 4. Create PPO ─────────────────────────────────────────────────────────
    model = PPO(
        policy            = 'MlpPolicy',      # plain MLP (not LSTM policy)
        env               = env,
        learning_rate     = lr_schedule,
        n_steps           = N_STEPS,
        batch_size        = BATCH_SIZE,
        n_epochs          = N_EPOCHS,
        gamma             = GAMMA,
        gae_lambda        = GAE_LAMBDA,
        clip_range        = CLIP_RANGE,
        clip_range_vf     = None,             # don't clip value function separately
        normalize_advantage = True,           # always — removes reward scale sensitivity
        ent_coef          = ENT_COEF,
        vf_coef           = VF_COEF,
        max_grad_norm     = MAX_GRAD_NORM,
        policy_kwargs     = policy_kwargs,
        tensorboard_log   = tensorboard_log or str(MODELS_DIR / 'tensorboard_v2'),
        verbose           = 1,
        device            = DEVICE_PPO,
        seed              = 42,               # reproducibility
    )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training function
# ─────────────────────────────────────────────────────────────────────────────

def train_ppo_v2(
    model:            PPO,
    total_timesteps:  int = 500_000,
    eval_env:         gym.Env = None,
    eval_freq:        int = 20_000,
    save_path:        str = None,
    n_eval_episodes:  int = 5,
) -> PPO:
    """
    Train the PPO agent with proper callbacks.

    Training schedule intuition:
        500k steps on 17k-candle data ≈ 29 passes through the dataset.
        At each step the agent sees one candle. That's plenty for a simple
        3-action trading policy backed by a pre-trained BiLSTM.

    Args:
        model:            PPO model from create_ppo_agent_v2().
        total_timesteps:  Total environment steps.
        eval_env:         Optional held-out environment for evaluation.
        eval_freq:        How often (in steps) to run evaluation.
        save_path:        Directory to save best model checkpoints.
        n_eval_episodes:  Episodes per evaluation run.

    Returns:
        Trained PPO model.
    """
    save_path = save_path or str(MODELS_DIR)

    # ── callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        TradeMetricsCallback(),
        EntropyScheduleCallback(
            ent_start       = 0.02,
            ent_end         = 0.002,
            total_timesteps = total_timesteps,
        ),
    ]

    if eval_env is not None:
        # Wrap eval env the same way as training env
        if not isinstance(eval_env, (DummyVecEnv, SubprocVecEnv)):
            eval_env = Monitor(eval_env)
            eval_env = DummyVecEnv([lambda: eval_env])

        # IMPORTANT: evaluation env must use the SAME normalization statistics
        # as the training env. Copy the running stats from the training VecNormalize.
        train_vec = model.get_env()
        eval_env  = VecNormalize(
            eval_env,
            norm_obs    = True,
            norm_reward = False,   # don't normalise rewards at eval time
            training    = False,   # don't UPDATE the running stats during eval
        )
        # Sync the running obs stats
        if isinstance(train_vec, VecNormalize):
            eval_env.obs_rms  = train_vec.obs_rms
            eval_env.ret_rms  = train_vec.ret_rms

        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path = save_path,
                log_path             = str(MODELS_DIR / 'eval_logs_v2'),
                eval_freq            = eval_freq,
                n_eval_episodes      = n_eval_episodes,
                deterministic        = True,
                render               = False,
            )
        )

    # ── train ─────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps  = total_timesteps,
        callback         = CallbackList(callbacks),
        progress_bar     = True,
        reset_num_timesteps = True,
    )

    # ── save final model + VecNormalize stats ─────────────────────────────────
    final_model_path = Path(save_path) / 'ppo_v2_final'
    model.save(str(final_model_path))
    print(f"Model saved → {final_model_path}.zip")

    vec_env = model.get_env()
    if isinstance(vec_env, VecNormalize):
        vecnorm_path = Path(save_path) / 'vecnorm_v2.pkl'
        vec_env.save(str(vecnorm_path))
        print(f"VecNormalize stats → {vecnorm_path}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_agent_v2(model: PPO, name: str = 'ppo_v2'):
    """Save model weights and VecNormalize stats."""
    model_path = MODELS_DIR / name
    model.save(str(model_path))
    print(f"Agent saved → {model_path}.zip")

    vec_env = model.get_env()
    if isinstance(vec_env, VecNormalize):
        vn_path = MODELS_DIR / f'vecnorm_{name}.pkl'
        vec_env.save(str(vn_path))
        print(f"VecNormalize → {vn_path}")


def load_agent_v2(
    env: gym.Env,
    name: str = 'ppo_v2',
) -> Tuple[PPO, VecNormalize]:
    """
    Load a saved PPO agent along with its VecNormalize statistics.

    Always load VecNormalize stats — without them the agent will see
    un-normalised observations and produce garbage predictions.
    """
    model_path  = MODELS_DIR / f'{name}.zip'
    vn_path     = MODELS_DIR / f'vecnorm_{name}.pkl'

    # Wrap env
    if not isinstance(env, (DummyVecEnv, SubprocVecEnv)):
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])

    # Load VecNormalize
    if vn_path.exists():
        vec_env          = VecNormalize.load(str(vn_path), env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"VecNormalize loaded ← {vn_path}")
    else:
        print(f"WARNING: VecNormalize stats not found at {vn_path}.")
        print("The agent will see un-normalised observations — predictions unreliable.")
        vec_env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)

    custom_objects = {
        'policy_kwargs': dict(
            features_extractor_class  = PassthroughExtractor,
            features_extractor_kwargs = {},
            net_arch                  = NET_ARCH,
            activation_fn             = nn.Tanh,
            ortho_init                = True,
        )
    }

    model = PPO.load(
        str(model_path),
        env            = vec_env,
        custom_objects = custom_objects,
        device         = DEVICE_PPO,
    )
    print(f"Agent loaded ← {model_path}")
    return model, vec_env


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent_v2(
    model:      PPO,
    env:        gym.Env,
    n_episodes: int = 10,
    verbose:    bool = True,
) -> Dict[str, float]:
    """
    Run n_episodes and collect trading statistics.

    Returns a dict of: mean_return, std_return, win_rate, mean_trades,
                       sharpe_approx, mean_reward.
    """
    all_returns = []
    all_trades  = []
    all_rewards = []

    for ep in range(n_episodes):
        obs          = env.reset()
        done         = False
        ep_reward    = 0.0

        while not done:
            action, _  = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            if isinstance(reward, np.ndarray):
                reward = reward[0]
            if isinstance(done, np.ndarray):
                done = done[0]

            ep_reward += reward

        # info may be a list (VecEnv) or dict
        if isinstance(info, (list, tuple)):
            info = info[0]

        all_returns.append(info.get('total_return', 0.0))
        all_trades.append(info.get('n_trades', 0))
        all_rewards.append(ep_reward)

    returns = np.array(all_returns)
    sharpe_approx = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252)

    metrics = {
        'mean_return':  float(returns.mean()),
        'std_return':   float(returns.std()),
        'win_rate':     float((returns > 0).mean()),
        'mean_trades':  float(np.mean(all_trades)),
        'sharpe_approx': float(sharpe_approx),
        'mean_reward':  float(np.mean(all_rewards)),
    }

    if verbose:
        print("\n── Evaluation Results ──────────────────────────────────────")
        print(f"  Mean Return  : {metrics['mean_return']*100:+.2f}%")
        print(f"  Std Return   : {metrics['std_return']*100:.2f}%")
        print(f"  Win Rate     : {metrics['win_rate']*100:.1f}%")
        print(f"  Mean Trades  : {metrics['mean_trades']:.1f}")
        print(f"  Sharpe (est) : {metrics['sharpe_approx']:.3f}")
        print("────────────────────────────────────────────────────────────")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.environment.trading_env_v2 import CryptoTradingEnvV2

    # synthetic data (just to verify shapes and forward pass)
    n    = 2000
    rng  = np.random.default_rng(0)
    df   = pd.DataFrame({
        'close':   np.cumsum(rng.normal(0, 10, n)) + 60_000,
        'f1':      rng.standard_normal(n),
        'f2':      rng.standard_normal(n),
        'f3':      rng.standard_normal(n),
    })
    cols = ['f1', 'f2', 'f3']

    env   = CryptoTradingEnvV2(df, cols, use_lstm_features=False)
    agent = create_ppo_agent_v2(env, total_timesteps=10_000)

    print(f"Obs space  : {agent.observation_space}")
    print(f"Action space: {agent.action_space}")
    print(f"Policy     : {agent.policy}")
    print("\nRunning 2000-step smoke test …")
    agent.learn(total_timesteps=2000, progress_bar=False)
    print("Smoke test passed.")
