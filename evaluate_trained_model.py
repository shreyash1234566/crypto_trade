"""
evaluate_trained_model.py - Evaluate the trained PPO model on the TEST set.

This runs the saved best_model.zip on the 15% held-out test data
to get the TRUE out-of-sample ROI.
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
from src.models.ppo_agent_v2 import evaluate_agent_v2, PassthroughExtractor, NET_ARCH

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import torch.nn as nn

DATA_PATH  = ROOT / 'btc_usdt_1h.csv'
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15

print("\n" + "="*60)
print("  STEP 1 - Load data and build features")
print("="*60)
raw = pd.read_csv(DATA_PATH)
df = build_features(raw)
feat_cols = [c for c in BILSTM_FEATURES if c in df.columns]
print(f"  {len(raw):,} rows, {len(feat_cols)} features")

print("\n" + "="*60)
print("  STEP 2 - Split data (same as training)")
print("="*60)
n = len(df)
train_end = int(n * TRAIN_FRAC)
val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
df_test = df.iloc[val_end:].reset_index(drop=True)
print(f"  Test set: {len(df_test):,} rows")

print("\n" + "="*60)
print("  STEP 3 - Load frozen BiLSTM")
print("="*60)
try:
    bilstm = load_bilstm('bilstm_best.pt')
    freeze_model(bilstm)
    use_lstm = True
    print("  BiLSTM loaded and frozen [OK]")
except FileNotFoundError:
    print("  WARNING: bilstm_best.pt not found.")
    bilstm = None
    use_lstm = False

print("\n" + "="*60)
print("  STEP 4 - Create test environment")
print("="*60)
env_kwargs = dict(
    feature_cols=feat_cols,
    initial_balance=10_000.0,
    transaction_fee=0.001,
    use_lstm_features=use_lstm,
    lstm_model=bilstm,
)
test_env = CryptoTradingEnvV3(df_test, **env_kwargs)
obs_sample, _ = test_env.reset()
print(f"  Obs shape: {obs_sample.shape}")

print("\n" + "="*60)
print("  STEP 5 - Load saved PPO model")
print("="*60)

# Try best_model first, then ppo_v2_final
model_path = MODELS_DIR / 'best_model.zip'
if not model_path.exists():
    model_path = MODELS_DIR / 'ppo_v2_final.zip'

vecnorm_path = MODELS_DIR / 'vecnorm_v2.pkl'

# Wrap test env
test_mon = Monitor(test_env)
test_vec = DummyVecEnv([lambda: test_mon])

# Load VecNormalize stats if available
if vecnorm_path.exists():
    test_vecnorm = VecNormalize.load(str(vecnorm_path), test_vec)
    test_vecnorm.training = False
    test_vecnorm.norm_reward = False
    print(f"  VecNormalize loaded from {vecnorm_path}")
else:
    test_vecnorm = VecNormalize(test_vec, norm_obs=True, norm_reward=False, training=False)
    print("  WARNING: No VecNormalize stats found, using fresh normalization")

custom_objects = {
    'policy_kwargs': dict(
        features_extractor_class=PassthroughExtractor,
        features_extractor_kwargs={},
        net_arch=NET_ARCH,
        activation_fn=nn.Tanh,
        ortho_init=True,
    )
}

model = PPO.load(
    str(model_path),
    env=test_vecnorm,
    custom_objects=custom_objects,
    device='cuda',
)
print(f"  Model loaded from {model_path}")

print("\n" + "="*60)
print("  STEP 6 - Evaluate on TEST set (5 episodes)")
print("="*60)
metrics = evaluate_agent_v2(model, test_vecnorm, n_episodes=5, verbose=True)

print("\n" + "="*60)
print("  FINAL RESULTS")
print("="*60)
roi = metrics['mean_return'] * 100
if roi > 0:
    print(f"  ROI: +{roi:.2f}% -- POSITIVE!")
else:
    print(f"  ROI: {roi:.2f}% -- NEGATIVE")
print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
print(f"  Mean Trades: {metrics['mean_trades']:.1f}")
print(f"  Sharpe Ratio: {metrics['sharpe_approx']:.3f}")
