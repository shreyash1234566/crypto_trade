#!/usr/bin/env python3
"""
PPO v4 Paper Trading Script
----------------------------------------------------------------------------
Loads the trained PPO v4 model (ppo_v4_final.zip + vecnorm_ppo_v4_final.pkl)
and runs it through CryptoTradingEnvV4 on the held-out test set.

Correct pipeline (mirrors train_v3.py):
  1. Load raw CSV -> build_features_v3()
  2. Split data -> use test set (last 15%)
  3. Load & freeze BiLSTM
  4. Create CryptoTradingEnvV4 with proper feature columns
  5. Load PPO v4 via load_agent_v2('ppo_v4_final')
  6. Run inference loop through VecNormalize
  7. Feed actions into PaperTrader execution simulator
  8. Save results as JSON for dashboard

Usage:
    python ppo_paper_trader.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import SEQUENCE_LENGTH, STATE_DIM, MODELS_DIR
from src.models.bilstm import load_model as load_bilstm, freeze_model
from src.data.features_v3 import build_features_v3, BILSTM_FEATURES, PPO_STRATEGY_FEATURES_V4
from src.environment.trading_env_v4 import CryptoTradingEnvV4
from src.models.ppo_agent_v2 import load_agent_v2
from src.execution.paper_trader import PaperTrader

DATA_PATH  = ROOT / 'btc_usdt_1h.csv'
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
LOG_DIR    = ROOT / 'logs'


def main():
    print("\n" + "=" * 65)
    print("  PPO v4 Paper Trading Session")
    print("=" * 65)

    # -- 1. Verify model files ------------------------------------------------
    model_path   = MODELS_DIR / 'ppo_v4_final.zip'
    vecnorm_path = MODELS_DIR / 'vecnorm_ppo_v4_final.pkl'

    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        print("   Run train_v3.py first to generate the model.")
        return
    if not vecnorm_path.exists():
        print(f"[ERROR] VecNormalize file not found: {vecnorm_path}")
        print("   Run train_v3.py first to generate the normalization stats.")
        return

    print(f"  [OK] Model:        {model_path.name}")
    print(f"  [OK] VecNormalize:  {vecnorm_path.name}")

    # -- 2. Load data & build features (v3 = v2 + 3 strategy signals) ---------
    print("\n" + "=" * 65)
    print("  STEP 1 -- Load data & build features_v3")
    print("=" * 65)
    raw = pd.read_csv(DATA_PATH)
    df  = build_features_v3(raw)
    feat_cols = [c for c in BILSTM_FEATURES if c in df.columns]
    print(f"  {len(raw):,} rows  |  {len(feat_cols)} BiLSTM features")
    print(f"  Strategy features: {len(PPO_STRATEGY_FEATURES_V4)} (18)")

    # -- 3. Use test split (last 15%) -- same split as train_v3.py ------------
    print("\n" + "=" * 65)
    print("  STEP 2 -- Extract test set (last 15%)")
    print("=" * 65)
    n       = len(df)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    df_test = df.iloc[val_end:].reset_index(drop=True)
    print(f"  Test set: {len(df_test):,} rows (out-of-sample)")

    # -- 4. Load frozen BiLSTM -------------------------------------------------
    print("\n" + "=" * 65)
    print("  STEP 3 -- Load frozen BiLSTM")
    print("=" * 65)
    try:
        bilstm = load_bilstm('bilstm_best.pt')
        freeze_model(bilstm)
        use_lstm = True
        print("  [OK] BiLSTM loaded and frozen")
    except FileNotFoundError:
        print("  [WARN] bilstm_best.pt not found -- continuing without BiLSTM")
        print("     (Run train_bilstm.py first for best results)")
        bilstm   = None
        use_lstm = False

    # -- 5. Create CryptoTradingEnvV4 ------------------------------------------
    print("\n" + "=" * 65)
    print("  STEP 4 -- Create CryptoTradingEnvV4")
    print("=" * 65)
    env_kwargs = dict(
        feature_cols      = feat_cols,
        initial_balance   = 10_000.0,
        transaction_fee   = 0.001,
        use_lstm_features = use_lstm,
        lstm_model        = bilstm,
    )
    test_env = CryptoTradingEnvV4(df_test, **env_kwargs)
    obs_sample, _ = test_env.reset()
    print(f"  Obs shape  : {obs_sample.shape}")
    print(f"    [0:{STATE_DIM}]   BiLSTM market state")
    print(f"    [{STATE_DIM}:{STATE_DIM+6}]  Account state")
    print(f"    [{STATE_DIM+6}:{STATE_DIM+6+len(PPO_STRATEGY_FEATURES_V4)}]  Strategy signals")
    print(f"  Actions    : 0=flat  1=long  2=short")

    # -- 6. Load PPO v4 model + VecNormalize stats -----------------------------
    print("\n" + "=" * 65)
    print("  STEP 5 -- Load PPO v4 + VecNormalize")
    print("=" * 65)
    model, vec_env = load_agent_v2(test_env, name='ppo_v4_final')
    print("  [OK] PPO v4 model loaded with correct VecNormalize stats")

    # -- 7. Run paper trading on full test set ---------------------------------
    print("\n" + "=" * 65)
    print("  STEP 6 -- Running paper trading inference")
    print("=" * 65)
    print(f"  {'Step':>6} | {'Price':>10} | {'Pos':>6} | {'Balance':>12} | {'Return':>8} | Trades")
    print("  " + "-" * 62)

    # Initialize PaperTrader for execution simulation
    trader = PaperTrader(initial_balance=10_000.0, use_testnet=True)

    obs        = vec_env.reset()
    done       = False
    step_count = 0
    action_names = {0: 'FLAT', 1: 'LONG', 2: 'SHORT'}

    # Collect step-by-step data for dashboard
    equity_curve = []
    action_dist  = {0: 0, 1: 0, 2: 0}

    while not done:
        # Get action from PPO model (through VecNormalize)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        # Unwrap VecEnv arrays
        if isinstance(done, np.ndarray):
            done = done[0]
        if isinstance(reward, np.ndarray):
            reward = reward[0]
        if isinstance(info, (list, tuple)):
            info = info[0]

        action_int = int(action[0]) if hasattr(action, '__len__') else int(action)
        action_dist[action_int] += 1
        step_count += 1

        # Extract info from the environment
        price    = info.get('current_price', 0.0)
        balance  = info.get('balance', 10_000.0)
        position = info.get('position', 0)

        # -- Feed action into PaperTrader execution simulator ------------------
        # Env actions: 0=flat, 1=long, 2=short
        # PaperTrader: 0=buy,  1=hold, 2=sell
        if action_int == 1 and trader.symbol not in trader.open_positions:
            # Go long -> buy
            trader.execute_signal(action=0, current_price=price,
                                  current_volatility=0.02)
        elif action_int == 0 and trader.symbol in trader.open_positions:
            # Go flat -> sell existing
            trader.execute_signal(action=2, current_price=price,
                                  current_volatility=0.02)
        elif action_int == 2 and trader.symbol in trader.open_positions:
            # Go short -> sell existing
            trader.execute_signal(action=2, current_price=price,
                                  current_volatility=0.02)

        # Check stop loss on open positions
        if trader.symbol in trader.open_positions:
            trader.check_stop_loss(price)

        # Log equity point
        equity_curve.append({
            'step':     step_count,
            'price':    round(price, 2),
            'balance':  round(balance, 2),
            'position': position,
            'action':   action_names.get(action_int, '?'),
            'reward':   round(float(reward), 6),
        })

        # Progress reporting every 500 steps + final
        if step_count % 500 == 0 or done:
            pos_str   = {0: ' FLAT', 1: '+LONG', -1: '-SHRT'}.get(position, '?')
            total_ret = info.get('total_return', 0.0) * 100
            n_trades  = info.get('n_trades', 0)
            print(
                f"  {step_count:6d} | "
                f"${price:>9,.2f} | {pos_str} | "
                f"${balance:>10,.2f} | "
                f"{total_ret:>+7.2f}% | {n_trades}"
            )

    # -- 8. Final results ------------------------------------------------------
    final_return  = info.get('total_return', 0.0)
    final_balance = info.get('balance', balance)
    n_trades      = info.get('n_trades', 0)
    avg_hold      = info.get('avg_hold', 0.0)

    print("\n" + "=" * 65)
    print("  PPO v4 PAPER TRADING RESULTS")
    print("=" * 65)
    print(f"  Model           : ppo_v4_final (PPO v2 architecture)")
    print(f"  Environment     : CryptoTradingEnvV4 (88-dim obs)")
    print(f"  Dataset         : Test set ({len(df_test):,} rows, out-of-sample)")
    print("-" * 65)
    print(f"  Initial Balance : $10,000.00")
    print(f"  Final Balance   : ${final_balance:,.2f}")
    roi = final_return * 100
    direction = "[UP]" if roi > 0 else "[DN]"
    print(f"  Total Return    : {direction} {roi:+.2f}%")
    print(f"  Total Trades    : {n_trades}")
    print(f"  Avg Hold (steps): {avg_hold:.1f}")
    print(f"  Total Steps     : {step_count}")
    print(f"  Action Dist     : Flat={action_dist[0]}  Long={action_dist[1]}  Short={action_dist[2]}")

    # Win rate from environment trades
    underlying_env = vec_env.venv.envs[0]
    if hasattr(underlying_env, 'env'):
        # Unwrap Monitor
        underlying_env = underlying_env.env
    trade_history = underlying_env.get_trade_history()
    if not trade_history.empty:
        wins = (trade_history['pnl_pct'] > 0).sum()
        losses = (trade_history['pnl_pct'] <= 0).sum()
        win_rate = wins / len(trade_history) * 100
        avg_win = trade_history.loc[trade_history['pnl_pct'] > 0, 'pnl_pct'].mean() * 100 \
                  if wins > 0 else 0
        avg_loss = trade_history.loc[trade_history['pnl_pct'] <= 0, 'pnl_pct'].mean() * 100 \
                   if losses > 0 else 0
        print(f"  Win Rate        : {win_rate:.1f}%  ({wins}W / {losses}L)")
        print(f"  Avg Win         : {avg_win:+.2f}%")
        print(f"  Avg Loss        : {avg_loss:+.2f}%")

    # PaperTrader execution summary
    print("\n  --- PaperTrader Execution Layer ---")
    trader.print_summary()

    # -- 9. Save results for dashboard ----------------------------------------
    LOG_DIR.mkdir(exist_ok=True)

    # Convert trade history to serializable format
    trade_list = []
    if not trade_history.empty:
        for _, row in trade_history.iterrows():
            trade_list.append({
                'direction':     int(row['direction']),
                'entry_price':   round(float(row['entry_price']), 2),
                'exit_price':    round(float(row['exit_price']), 2),
                'pnl_pct':       round(float(row['pnl_pct']) * 100, 4),
                'hold_duration': int(row['hold_duration']),
                'reason':        str(row['reason']),
            })

    results = {
        'timestamp':       datetime.now().isoformat(),
        'model':           'ppo_v4_final',
        'environment':     'CryptoTradingEnvV4',
        'dataset':         f'test set ({len(df_test):,} rows)',
        'initial_balance': 10_000.0,
        'final_balance':   round(final_balance, 2),
        'total_return_pct': round(roi, 4),
        'n_trades':        n_trades,
        'total_steps':     step_count,
        'action_dist':     {str(k): v for k, v in action_dist.items()},
        'equity_curve':    equity_curve,
        'trades':          trade_list,
    }

    results_path = LOG_DIR / 'paper_trade_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  [SAVED] Results saved -> {results_path}")

    # Save PaperTrader logs
    trader.save_trades()
    trader.save_metrics()

    print("\n  Paper trading session completed!")
    return results


if __name__ == '__main__':
    main()