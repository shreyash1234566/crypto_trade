"""
train_v3.py -- BiLSTM + PPO trained with all 3 static strategy signals.

Run AFTER train_bilstm.py:
    python train_bilstm.py   <- trains BiLSTM, saves bilstm_best.pt
    python train_v3.py       <- trains PPO with 3-strategy reward shaping

What's new vs train_v2.py:
-----------------------------------------------------------------------------
  1. Uses features_v3.build_features_v3()
     -> adds 12 new strategy signal columns to the DataFrame

  2. Uses trading_env_v4.CryptoTradingEnvV4
     -> obs is 88-dim (was 76): 64 BiLSTM + 6 account + 18 strategy
     -> 6 new reward signals from bb_rsi, macd_adx, and mic strategies

  3. Model saved as ppo_v4_final.zip

Flow:
  Data
  -> features_v3 (v2 features + 12 new strategy signals)
  -> BiLSTM (frozen) -> 64-dim market state
  -> CryptoTradingEnvV4 (88-dim obs, richer reward)
  -> PPO (ppo_agent_v2 architecture, PassthroughExtractor, 88-dim input)
  -> learns when each static strategy's signals are actually profitable
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import SEQUENCE_LENGTH, STATE_DIM, MODELS_DIR
from src.models.bilstm import load_model as load_bilstm, freeze_model
from src.data.features_v3 import build_features_v3, BILSTM_FEATURES, PPO_STRATEGY_FEATURES_V4
from src.environment.trading_env_v4 import CryptoTradingEnvV4
from src.models.ppo_agent_v2 import (
    create_ppo_agent_v2, train_ppo_v2, evaluate_agent_v2, save_agent_v2
)

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

DATA_PATH       = ROOT / 'btc_usdt_1h.csv'
TOTAL_TIMESTEPS = 500_000
EVAL_FREQ       = 25_000
N_EVAL_EPISODES = 5
TRAIN_FRAC      = 0.70
VAL_FRAC        = 0.15


# -----------------------------------------------------------------------------

print("\n" + "="*65)
print("  STEP 1 -- Load data")
print("="*65)
raw = pd.read_csv(DATA_PATH)
print(f"  {len(raw):,} rows  |  columns: {list(raw.columns)}")


print("\n" + "="*65)
print("  STEP 2 -- Build features (v3 = v2 + 3 strategy signals)")
print("="*65)
df = build_features_v3(raw)
feat_cols = [c for c in BILSTM_FEATURES if c in df.columns]
print(f"\n  BiLSTM input features   : {len(feat_cols)}")
print(f"  PPO strategy features   : {len(PPO_STRATEGY_FEATURES_V4)}  (was 6 in v3)")
print(f"\n  Strategy feature breakdown:")
print(f"    [REGIME]   price_above_200, ema8_21_cross, price_above_ema21")
print(f"    [BB-RSI]   bb_oversold, rsi_oversold, stoch_oversold,")
print(f"               volume_spike, reversal_candle, bb_at_upper,")
print(f"               stoch_overbought, bbrsi_setup")
print(f"    [MACD-ADX] macd_just_crossed_up, macd_just_crossed_dn,")
print(f"               strong_trend, di_bull, macd_adx_setup")
print(f"    [MIC]      mic_setup, ema8_34_cross")


print("\n" + "="*65)
print("  STEP 3 -- Time-ordered split 70/15/15")
print("="*65)
n         = len(df)
train_end = int(n * TRAIN_FRAC)
val_end   = int(n * (TRAIN_FRAC + VAL_FRAC))
df_train  = df.iloc[:train_end].reset_index(drop=True)
df_val    = df.iloc[train_end:val_end].reset_index(drop=True)
df_test   = df.iloc[val_end:].reset_index(drop=True)
print(f"  Train : {len(df_train):,} rows")
print(f"  Val   : {len(df_val):,} rows")
print(f"  Test  : {len(df_test):,} rows")

# Verify strategy signals are present
for sig in ['bbrsi_setup', 'macd_adx_setup', 'mic_setup']:
    n_hits = int(df_train[sig].sum())
    pct    = n_hits / len(df_train) * 100
    print(f"  {sig:25s} -> {n_hits:,} setups in train ({pct:.1f}% of bars)")


print("\n" + "="*65)
print("  STEP 4 -- Load frozen BiLSTM")
print("="*65)
print("  NOTE: BiLSTM should be retrained with --weight-scale 0.5 before")
print("  running this script. The default weight causes too many false")
print("  positives (PR-AUC ~0.18, near-random). Correct command:")
print("    python train_bilstm.py --weight-scale 0.5")
print()
try:
    bilstm = load_bilstm('bilstm_best.pt')
    freeze_model(bilstm)
    use_lstm = True
    print("  BiLSTM loaded and frozen [OK]")
except FileNotFoundError:
    print("  WARNING: bilstm_best.pt not found.")
    print("  Run:  python train_bilstm.py")
    print("  Continuing with raw features (BiLSTM disabled).")
    bilstm   = None
    use_lstm = False


print("\n" + "="*65)
print("  STEP 5 -- Create CryptoTradingEnvV4 environments")
print("="*65)
env_kwargs = dict(
    feature_cols      = feat_cols,
    initial_balance   = 10_000.0,
    transaction_fee   = 0.001,
    use_lstm_features = use_lstm,
    lstm_model        = bilstm,
)
train_env = CryptoTradingEnvV4(df_train, **env_kwargs)
val_env   = CryptoTradingEnvV4(df_val,   **env_kwargs)
test_env  = CryptoTradingEnvV4(df_test,  **env_kwargs)

obs_sample, _ = train_env.reset()
print(f"  Obs shape  : {obs_sample.shape}")
print(f"    [0:64]   BiLSTM market state (60-candle compressed pattern)")
print(f"    [64:70]  Account state (position, PnL, balance, steps held)")
print(f"    [70:88]  Strategy signals (18 features from 3 static strategies)")
print(f"  Action space: 0=flat  1=long  2=short")

print("\n  Reward signals in v4:")
print("  -- Carried from v3 ----------------------------------")
print("    -0.005/step  regime penalty  (long below EMA200)")
print("    -0.003/step  trailing penalty (long below EMA21)")
print("    +0.003       EMA cross entry bonus")
print("    -0.006       churn penalty  (trade < 12 steps)")
print("    -0.002 x dd  drawdown penalty (scaled by current drawdown)")
print("  -- New from 3 static strategies ---------------------")
print("    +0.005       MIC confluence bonus    (all indicators agree)")
print("    +0.004       BB-RSI entry bonus      (oversold bounce setup)")
print("    +0.003       MACD-ADX entry bonus    (fresh cross + trend)")
print("    -0.003/step  BB overbought penalty   (bb_pct>0.85 or RSI>65)")
print("    -0.003/step  trend death penalty     (MACD dn or DI bearish)")
print("    -0.002/step  stoch overbought penalty(stoch_k > 80)")


print("\n" + "="*65)
print("  STEP 6 -- Create PPO agent  (88-dim obs auto-detected)")
print("="*65)
agent = create_ppo_agent_v2(train_env, total_timesteps=TOTAL_TIMESTEPS)
total_params = sum(p.numel() for p in agent.policy.parameters())
obs_dim      = obs_sample.shape[0]
print(f"  Policy params : {total_params:,}")
print(f"  Obs dim       : {obs_dim}  (PassthroughExtractor auto-detects)")
print(f"  Net arch      : pi=[128,64,32]  vf=[128,64,32]  activation=Tanh")


print("\n" + "="*65)
print(f"  STEP 7 -- Train {TOTAL_TIMESTEPS:,} steps")
print("="*65)
print("  TensorBoard: tensorboard --logdir models_saved/tensorboard_v2\n")

agent = train_ppo_v2(
    model           = agent,
    total_timesteps = TOTAL_TIMESTEPS,
    eval_env        = val_env,
    eval_freq       = EVAL_FREQ,
    n_eval_episodes = N_EVAL_EPISODES,
    save_path       = str(MODELS_DIR),
)


print("\n" + "="*65)
print("  STEP 8 -- Test set evaluation")
print("="*65)
test_mon     = Monitor(test_env)
test_vec     = DummyVecEnv([lambda: test_mon])
train_vec    = agent.get_env()
test_vecnorm = VecNormalize(test_vec, norm_obs=True, norm_reward=False, training=False)
if isinstance(train_vec, VecNormalize):
    test_vecnorm.obs_rms = train_vec.obs_rms

metrics = evaluate_agent_v2(agent, test_vecnorm, n_episodes=N_EVAL_EPISODES)
print(f"\n  Return   : {metrics['mean_return']*100:+.2f}% +/- {metrics['std_return']*100:.2f}%")
print(f"  Win Rate : {metrics['win_rate']*100:.1f}%")
print(f"  Trades   : {metrics['mean_trades']:.1f}")
print(f"  Sharpe   : {metrics['sharpe_approx']:.3f}")


print("\n" + "="*65)
print("  STEP 9 -- Save")
print("="*65)
save_agent_v2(agent, name='ppo_v4_final')
print("\n  Files saved:")
print("    models_saved/ppo_v4_final.zip")
print("    models_saved/vecnorm_ppo_v4_final.pkl")
print("\n  To evaluate later:")
print("    from src.models.ppo_agent_v2 import load_agent_v2")
print("    model, vec_env = load_agent_v2(env, name='ppo_v4_final')")
