"""
train_v2.py — End-to-end training script for Bi-LSTM + PPO v2.

Run this file to train the corrected system:
    python train_v2.py

What this script does (in order):
  1. Load your real BTC/USDT 1h data
  2. Build features (the same feature engineering pipeline)
  3. Load the pre-trained frozen BiLSTM
  4. Split data: 70% train / 15% val / 15% test  (time-ordered, no shuffle)
  5. Create CryptoTradingEnvV2 for train + val
  6. Create PPO agent v2 with correct hyperparameters
  7. Train for 500k steps
  8. Evaluate on held-out test set
  9. Save model + VecNormalize stats
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── make imports work from project root ──────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config.settings import (
    SEQUENCE_LENGTH, STATE_DIM, MODELS_DIR,
    FEATURE_COLUMNS
)
from src.models.bilstm import load_model as load_bilstm, freeze_model
from src.data.features import build_features           # your existing pipeline

# ── NEW v2 imports ────────────────────────────────────────────────────────────
from trading_env_v2   import CryptoTradingEnvV2, make_env_v2
from ppo_agent_v2     import (
    create_ppo_agent_v2, train_ppo_v2,
    evaluate_agent_v2, save_agent_v2
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH        = ROOT / 'btc_usdt_1h.csv'
TOTAL_TIMESTEPS  = 500_000   # total PPO training steps
EVAL_FREQ        = 25_000    # how often to run evaluation callback
N_EVAL_EPISODES  = 5         # episodes per evaluation

TRAIN_FRAC       = 0.70      # 70% of data for training
VAL_FRAC         = 0.15      # 15% for validation (PPO callback)
# remaining 15%  = test (never seen during training)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & prepare data
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  STEP 1 — Loading data")
print("═"*60)

raw_df = pd.read_csv(DATA_PATH)
print(f"  Raw data : {raw_df.shape[0]:,} rows  ({raw_df.shape[0]//24} days)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  STEP 2 — Feature engineering")
print("═"*60)

try:
    df = build_features(raw_df)
    print(f"  Features built : {df.shape[1]} columns")
except Exception as e:
    print(f"  WARNING: build_features() failed ({e})")
    print("  Falling back to raw OHLCV features.")
    df = raw_df.copy()

# Find which feature columns are actually present
available_feats = [c for c in FEATURE_COLUMNS if c in df.columns]
if not available_feats:
    # Absolute fallback: use whatever numeric columns exist (minus close)
    available_feats = [c for c in df.select_dtypes(include='number').columns
                       if c not in ('open','high','low','volume','target')]

print(f"  Feature columns : {len(available_feats)} → {available_feats[:5]} …")

# Ensure 'close' is in the dataframe
assert 'close' in df.columns, "DataFrame must have a 'close' column"
df = df.dropna(subset=['close']).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Train / Val / Test split   (time-ordered — NO shuffle)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  STEP 3 — Time-ordered train/val/test split")
print("═"*60)

n          = len(df)
train_end  = int(n * TRAIN_FRAC)
val_end    = int(n * (TRAIN_FRAC + VAL_FRAC))

df_train   = df.iloc[:train_end].reset_index(drop=True)
df_val     = df.iloc[train_end:val_end].reset_index(drop=True)
df_test    = df.iloc[val_end:].reset_index(drop=True)

print(f"  Train : {len(df_train):>6,} rows  ({len(df_train)//24:>4} days)")
print(f"  Val   : {len(df_val):>6,} rows  ({len(df_val)//24:>4} days)")
print(f"  Test  : {len(df_test):>6,} rows  ({len(df_test)//24:>4} days)")

# Sanity: val/test must have enough candles for at least one LSTM window
assert len(df_val)  > SEQUENCE_LENGTH * 2, "Val set too small for LSTM sequences"
assert len(df_test) > SEQUENCE_LENGTH * 2, "Test set too small for LSTM sequences"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Load frozen BiLSTM
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  STEP 4 — Loading frozen BiLSTM")
print("═"*60)

try:
    bilstm = load_bilstm('bilstm_best.pt')
    freeze_model(bilstm)
    use_lstm = True
    print("  BiLSTM loaded and frozen ✓")
    print(f"  Output state dim : {STATE_DIM}")
except FileNotFoundError:
    print("  WARNING: bilstm_best.pt not found.")
    print("  The PPO will train on raw features instead of BiLSTM states.")
    print("  Run train_bilstm_refined.py first for best results.")
    bilstm   = None
    use_lstm = False


# ─────────────────────────────────────────────────────────────────────────────
# 5. Create environments
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  STEP 5 — Creating environments")
print("═"*60)

env_kwargs = dict(
    feature_cols      = available_feats,
    initial_balance   = 10_000.0,
    transaction_fee   = 0.001,        # 0.1% Binance fee
    use_lstm_features = use_lstm,
    lstm_model        = bilstm,
)

train_env = CryptoTradingEnvV2(df_train, **env_kwargs)
val_env   = CryptoTradingEnvV2(df_val,   **env_kwargs)
test_env  = CryptoTradingEnvV2(df_test,  **env_kwargs)

print(f"  Train env obs shape : {train_env.observation_space.shape}")
print(f"  Actions             : {train_env.action_space.n}  (0=flat, 1=long, 2=short)")
obs_sample, _ = train_env.reset()
print(f"  Sample obs[:5]      : {obs_sample[:5].round(4)}")
print(f"  Sample obs[64:]     : {obs_sample[64:].round(4)}  ← account features")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Create PPO agent
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  STEP 6 — Creating PPO agent v2")
print("═"*60)

agent = create_ppo_agent_v2(
    env             = train_env,
    total_timesteps = TOTAL_TIMESTEPS,
)

# Print architecture for transparency
total_params = sum(p.numel() for p in agent.policy.parameters())
print(f"  Policy params    : {total_params:,}")
print(f"  Device           : {agent.device}")
print(f"  Hyperparameters:")
print(f"    n_steps        : {agent.n_steps}")
print(f"    batch_size     : {agent.batch_size}")
print(f"    n_epochs       : {agent.n_epochs}")
print(f"    gamma          : {agent.gamma}")
print(f"    gae_lambda     : {agent.gae_lambda}")
print(f"    clip_range     : {agent.clip_range}")
print(f"    ent_coef       : {agent.ent_coef}")
print(f"    vf_coef        : {agent.vf_coef}")
print(f"    max_grad_norm  : {agent.max_grad_norm}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Train
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print(f"  STEP 7 — Training for {TOTAL_TIMESTEPS:,} steps")
print("═"*60)
print("  Watch tensorboard: tensorboard --logdir models_saved/tensorboard_v2\n")

agent = train_ppo_v2(
    model           = agent,
    total_timesteps = TOTAL_TIMESTEPS,
    eval_env        = val_env,
    eval_freq       = EVAL_FREQ,
    n_eval_episodes = N_EVAL_EPISODES,
    save_path       = str(MODELS_DIR),
)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Evaluate on held-out test set
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  STEP 8 — Test set evaluation (never seen during training)")
print("═"*60)

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

test_vec = Monitor(test_env)
test_vec = DummyVecEnv([lambda: test_vec])

# Sync normalisation stats from training (IMPORTANT)
train_vec = agent.get_env()
test_vecnorm = VecNormalize(test_vec, norm_obs=True, norm_reward=False, training=False)
if isinstance(train_vec, VecNormalize):
    test_vecnorm.obs_rms = train_vec.obs_rms
    test_vecnorm.ret_rms = train_vec.ret_rms

metrics = evaluate_agent_v2(agent, test_vecnorm, n_episodes=N_EVAL_EPISODES)

print("\n  ── Final Test Metrics ─────────────────────────────────")
print(f"  Return  : {metrics['mean_return']*100:+.2f}%  ± {metrics['std_return']*100:.2f}%")
print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
print(f"  Trades  : {metrics['mean_trades']:.1f} per episode")
print(f"  Sharpe  : {metrics['sharpe_approx']:.3f}")
print("  ───────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Save
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  STEP 9 — Saving")
print("═"*60)

save_agent_v2(agent, name='ppo_v2_trained')
print("\n  Done. Files saved to models_saved/")
print("  ppo_v2_trained.zip")
print("  vecnorm_ppo_v2_trained.pkl")
