"""
train_v2.py — End-to-end: BiLSTM + PPO trained on +7.96% strategy signals.

Run AFTER train_bilstm.py:
    python train_bilstm.py   ← trains BiLSTM, saves bilstm_best.pt
    python train_v2.py       ← trains PPO on top, saves ppo_v3_final.zip

What the PPO learns from the +7.96% strategy:
  - Never go long when price < EMA200  (regime penalty in reward)
  - Exit when price drops below EMA21  (trailing stop penalty)
  - Enter on EMA8×EMA21 cross + regime (entry bonus in reward)
  - Win rate 42% is OK if wins are 2.5× bigger than losses
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import SEQUENCE_LENGTH, STATE_DIM, MODELS_DIR
from src.models.bilstm import load_model as load_bilstm, freeze_model
from src.data.features_v2 import build_features, BILSTM_FEATURES, PPO_STRATEGY_FEATURES
from src.environment.trading_env_v3 import CryptoTradingEnvV3
from src.models.ppo_agent_v2 import create_ppo_agent_v2, train_ppo_v2, evaluate_agent_v2, save_agent_v2

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

DATA_PATH       = ROOT / 'btc_usdt_1h.csv'
TOTAL_TIMESTEPS = 500_000
EVAL_FREQ       = 25_000
N_EVAL_EPISODES = 5
TRAIN_FRAC      = 0.70
VAL_FRAC        = 0.15


print("\n" + "="*60)
print("  STEP 1 — Load data")
print("="*60)
raw = pd.read_csv(DATA_PATH)
print(f"  {len(raw):,} rows")


print("\n" + "="*60)
print("  STEP 2 — Build features")
print("="*60)
df = build_features(raw)
feat_cols = [c for c in BILSTM_FEATURES if c in df.columns]
print(f"  {len(feat_cols)} BiLSTM features  |  {len(PPO_STRATEGY_FEATURES)} PPO strategy features")


print("\n" + "="*60)
print("  STEP 3 — Time-ordered split 70/15/15")
print("="*60)
n         = len(df)
train_end = int(n * TRAIN_FRAC)
val_end   = int(n * (TRAIN_FRAC + VAL_FRAC))
df_train  = df.iloc[:train_end].reset_index(drop=True)
df_val    = df.iloc[train_end:val_end].reset_index(drop=True)
df_test   = df.iloc[val_end:].reset_index(drop=True)
print(f"  Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")


print("\n" + "="*60)
print("  STEP 4 — Load frozen BiLSTM")
print("="*60)
try:
    bilstm = load_bilstm('bilstm_best.pt')
    freeze_model(bilstm)
    use_lstm = True
    print("  BiLSTM loaded and frozen ✓")
except FileNotFoundError:
    print("  WARNING: bilstm_best.pt not found. Run train_bilstm.py first.")
    print("  Continuing with raw features (no BiLSTM).")
    bilstm   = None
    use_lstm = False


print("\n" + "="*60)
print("  STEP 5 — Create environments")
print("="*60)
env_kwargs = dict(
    feature_cols      = feat_cols,
    initial_balance   = 10_000.0,
    transaction_fee   = 0.001,
    use_lstm_features = use_lstm,
    lstm_model        = bilstm,
)
train_env = CryptoTradingEnvV3(df_train, **env_kwargs)
val_env   = CryptoTradingEnvV3(df_val,   **env_kwargs)
test_env  = CryptoTradingEnvV3(df_test,  **env_kwargs)

obs_sample, _ = train_env.reset()
print(f"  Obs shape: {obs_sample.shape}  (64 BiLSTM + 6 account + 6 strategy)")
print(f"  Strategy features visible to PPO: {PPO_STRATEGY_FEATURES}")
print(f"  Actions: 0=flat  1=long  2=short")


print("\n" + "="*60)
print("  STEP 6 — Create PPO agent")
print("="*60)
agent = create_ppo_agent_v2(train_env, total_timesteps=TOTAL_TIMESTEPS)
total_params = sum(p.numel() for p in agent.policy.parameters())
print(f"  Policy params: {total_params:,}")


print("\n" + "="*60)
print(f"  STEP 7 — Train {TOTAL_TIMESTEPS:,} steps")
print("="*60)
print("  TensorBoard: tensorboard --logdir models_saved/tensorboard_v2\n")

agent = train_ppo_v2(
    model           = agent,
    total_timesteps = TOTAL_TIMESTEPS,
    eval_env        = val_env,
    eval_freq       = EVAL_FREQ,
    n_eval_episodes = N_EVAL_EPISODES,
    save_path       = str(MODELS_DIR),
)


print("\n" + "="*60)
print("  STEP 8 — Test set evaluation")
print("="*60)
test_mon     = Monitor(test_env)
test_vec     = DummyVecEnv([lambda: test_mon])
train_vec    = agent.get_env()
test_vecnorm = VecNormalize(test_vec, norm_obs=True, norm_reward=False, training=False)
if isinstance(train_vec, VecNormalize):
    test_vecnorm.obs_rms = train_vec.obs_rms

metrics = evaluate_agent_v2(agent, test_vecnorm, n_episodes=N_EVAL_EPISODES)
print(f"\n  Return:   {metrics['mean_return']*100:+.2f}% ± {metrics['std_return']*100:.2f}%")
print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
print(f"  Trades:   {metrics['mean_trades']:.1f}")
print(f"  Sharpe:   {metrics['sharpe_approx']:.3f}")


print("\n" + "="*60)
print("  STEP 9 — Save")
print("="*60)
save_agent_v2(agent, name='ppo_v3_trained')
print("\n  Files saved:")
print("  models_saved/ppo_v3_trained.zip")
print("  models_saved/vecnorm_ppo_v3_trained.pkl")
