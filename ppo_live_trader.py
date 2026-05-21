#!/usr/bin/env python3
"""
PPO v4 LIVE Paper Trading Script
----------------------------------------------------------------------------
Fetches real-time 1h candles from Binance, builds features, and runs the
trained PPO v4 model in a live loop. Trades are executed through the
PaperTrader execution simulator.

How it works:
  1. Fetch ~500 recent 1h candles from Binance for indicator warm-up
  2. Build features_v3 on the warm-up data
  3. Load frozen BiLSTM + PPO v4 model with VecNormalize
  4. Fast-forward through warm-up candles (model sees real market history)
  5. Poll Binance every 30s for new closed 1h candles
  6. On each new candle: rebuild features, step the model, execute trades
  7. Display live dashboard in terminal

Usage:
    python ppo_live_trader.py
    (Ctrl+C to stop)
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import ccxt
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import SEQUENCE_LENGTH, STATE_DIM, MODELS_DIR, SYMBOL
from src.models.bilstm import load_model as load_bilstm, freeze_model
from src.data.features_v3 import build_features_v3, BILSTM_FEATURES, PPO_STRATEGY_FEATURES_V4
from src.environment.trading_env_v4 import CryptoTradingEnvV4
from src.models.ppo_agent_v2 import load_agent_v2
from src.execution.paper_trader import PaperTrader

LOG_DIR        = ROOT / 'logs'
WARMUP_CANDLES = 500          # enough for EMA200 + sequence padding
POLL_INTERVAL  = 30           # seconds between Binance polls
INITIAL_BALANCE = 10_000.0


def clear_line():
    """Clear current terminal line."""
    print('\r' + ' ' * 120 + '\r', end='', flush=True)


def fetch_historical_candles(exchange, symbol, timeframe='1h', limit=500):
    """Fetch recent historical candles from Binance for warm-up."""
    print(f"  Fetching {limit} recent {timeframe} candles from Binance...")
    all_candles = []
    since = None
    remaining = limit

    while remaining > 0:
        batch_size = min(remaining, 1000)
        candles = exchange.fetch_ohlcv(
            symbol, timeframe, since=since, limit=batch_size
        )
        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + 1
        remaining -= len(candles)
        if len(candles) < batch_size:
            break
        time.sleep(0.2)

    if len(all_candles) > limit:
        all_candles = all_candles[-limit:]

    df = pd.DataFrame(
        all_candles,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df


def fetch_latest_closed_candle(exchange, symbol, timeframe='1h'):
    """Fetch the most recently CLOSED candle."""
    candles = exchange.fetch_ohlcv(symbol, timeframe, limit=2)
    if len(candles) < 2:
        return None
    # candles[-1] is the current (still open) candle
    # candles[-2] is the last fully CLOSED candle
    last_closed = candles[-2]
    return {
        'timestamp': pd.Timestamp(last_closed[0], unit='ms'),
        'open':      last_closed[1],
        'high':      last_closed[2],
        'low':       last_closed[3],
        'close':     last_closed[4],
        'volume':    last_closed[5],
    }


def connect_binance():
    """Connect to Binance with Vision API fallback."""
    # Try Vision API first (geo-unrestricted)
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        exchange.urls['api']['public'] = 'https://data-api.binance.vision/api/v3'
        ticker = exchange.fetch_ticker(SYMBOL)
        print(f"  [OK] Binance Vision API connected. {SYMBOL}: ${ticker['last']:,.2f}")
        return exchange
    except Exception as e:
        print(f"  Vision API failed: {e}")

    # Fallback: standard API
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        ticker = exchange.fetch_ticker(SYMBOL)
        print(f"  [OK] Binance API connected. {SYMBOL}: ${ticker['last']:,.2f}")
        return exchange
    except Exception as e:
        print(f"  [FAIL] Could not connect to Binance: {e}")
        sys.exit(1)


def print_dashboard(step, price, position, balance, total_return, n_trades,
                    action_name, last_signal, trader_pnl, next_check):
    """Print a live dashboard to the terminal."""
    pos_str = {0: 'FLAT', 1: 'LONG', -1: 'SHORT'}.get(position, '?')
    pos_marker = {0: '  ', 1: '+ ', -1: '- '}.get(position, '  ')
    now = datetime.now().strftime('%H:%M:%S')

    print()
    print("=" * 70)
    print(f"  PPO v4 LIVE PAPER TRADER              {now}")
    print("=" * 70)
    print(f"  BTC/USDT Price  : ${price:>12,.2f}")
    print(f"  Position        : {pos_marker}{pos_str}")
    print(f"  Last Action     : {action_name:<10s}  Signal: {last_signal}")
    print("-" * 70)
    print(f"  Balance         : ${balance:>12,.2f}")
    ret_dir = '[UP]' if total_return >= 0 else '[DN]'
    print(f"  Model Return    : {ret_dir} {total_return:+.2f}%")
    print(f"  Trades Executed : {n_trades}")
    print(f"  PaperTrader PnL : ${trader_pnl:>12,.2f}")
    print("-" * 70)
    print(f"  Live Candles    : {step}")
    print(f"  Next candle     : ~{next_check}")
    print("=" * 70)


def main():
    print("\n" + "=" * 70)
    print("  PPO v4 LIVE Paper Trading Session")
    print("=" * 70)
    print(f"  Symbol: {SYMBOL}  |  Timeframe: 1h  |  Poll: {POLL_INTERVAL}s")
    print(f"  Balance: ${INITIAL_BALANCE:,.2f}  |  Press Ctrl+C to stop")
    print("=" * 70)

    # -- 1. Verify model files ------------------------------------------------
    model_path   = MODELS_DIR / 'ppo_v4_final.zip'
    vecnorm_path = MODELS_DIR / 'vecnorm_ppo_v4_final.pkl'

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return
    if not vecnorm_path.exists():
        print(f"[ERROR] VecNormalize not found: {vecnorm_path}")
        return
    print(f"  [OK] Model: {model_path.name}")
    print(f"  [OK] VecNormalize: {vecnorm_path.name}")

    # -- 2. Connect to Binance ------------------------------------------------
    print("\n--- Connecting to Binance ---")
    exchange = connect_binance()

    # -- 3. Fetch warm-up candles ---------------------------------------------
    print("\n--- Fetching warm-up data ---")
    raw_df = fetch_historical_candles(exchange, SYMBOL, '1h', WARMUP_CANDLES)
    raw_df = raw_df.reset_index()
    raw_df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    print(f"  Fetched {len(raw_df)} candles")
    print(f"  From: {raw_df['timestamp'].iloc[0]}")
    print(f"  To:   {raw_df['timestamp'].iloc[-1]}")
    last_candle_ts = raw_df['timestamp'].iloc[-1]

    # -- 4. Build features ----------------------------------------------------
    print("\n--- Building features_v3 ---")
    df = build_features_v3(raw_df.copy())
    feat_cols = [c for c in BILSTM_FEATURES if c in df.columns]
    print(f"  {len(df)} rows | {len(feat_cols)} BiLSTM features | 18 strategy features")

    # -- 5. Load frozen BiLSTM ------------------------------------------------
    print("\n--- Loading BiLSTM ---")
    try:
        bilstm = load_bilstm('bilstm_best.pt')
        freeze_model(bilstm)
        use_lstm = True
        print("  [OK] BiLSTM loaded and frozen")
    except FileNotFoundError:
        print("  [WARN] BiLSTM not found, continuing without it")
        bilstm = None
        use_lstm = False

    # -- 6. Create environment with warm-up data ------------------------------
    print("\n--- Creating environment ---")
    env_kwargs = dict(
        feature_cols      = feat_cols,
        initial_balance   = INITIAL_BALANCE,
        transaction_fee   = 0.001,
        use_lstm_features = use_lstm,
        lstm_model        = bilstm,
    )
    env = CryptoTradingEnvV4(df, **env_kwargs)

    # -- 7. Load PPO model + VecNormalize -------------------------------------
    print("\n--- Loading PPO v4 model ---")
    model, vec_env = load_agent_v2(env, name='ppo_v4_final')
    print("  [OK] PPO v4 loaded with VecNormalize")

    # -- 8. Fast-forward through warm-up data ---------------------------------
    print("\n--- Fast-forwarding through warm-up candles ---")
    obs = vec_env.reset()
    warmup_steps = len(df) - SEQUENCE_LENGTH - 2  # leave last candle for live
    action_names = {0: 'FLAT', 1: 'LONG', 2: 'SHORT'}

    for i in range(warmup_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        if isinstance(done, np.ndarray):
            done = done[0]
        if done:
            break
        if (i + 1) % 100 == 0:
            clear_line()
            pct = (i + 1) / warmup_steps * 100
            print(f'\r  Warming up... {i+1}/{warmup_steps} ({pct:.0f}%)', end='', flush=True)

    # Get current state after warm-up
    if isinstance(info, (list, tuple)):
        info = info[0]
    warmup_balance = info.get('balance', INITIAL_BALANCE)
    warmup_trades  = info.get('n_trades', 0)
    warmup_pos     = info.get('position', 0)
    print(f"\n  Warm-up complete: {warmup_steps} candles processed")
    print(f"  Balance after warm-up: ${warmup_balance:,.2f}")
    print(f"  Trades during warm-up: {warmup_trades}")
    pos_name = {0: 'FLAT', 1: 'LONG', -1: 'SHORT'}.get(warmup_pos, '?')
    print(f"  Current position: {pos_name}")

    # -- 9. Initialize PaperTrader for live execution -------------------------
    trader = PaperTrader(initial_balance=INITIAL_BALANCE, use_testnet=True)

    # -- 10. LIVE LOOP --------------------------------------------------------
    print("\n" + "=" * 70)
    print("  >>> LIVE TRADING STARTED <<<")
    print("  Waiting for the next closed 1h candle...")
    print("=" * 70)

    live_step      = 0
    action_dist    = {0: 0, 1: 0, 2: 0}
    equity_curve   = []
    last_action    = 'FLAT'
    last_signal    = '--'
    balance        = warmup_balance
    total_ret      = (warmup_balance / INITIAL_BALANCE - 1.0) * 100
    n_trades       = warmup_trades
    position       = warmup_pos
    price          = float(df['close'].iloc[-1])

    try:
        while True:
            # Check for new closed candle
            try:
                latest = fetch_latest_closed_candle(exchange, SYMBOL, '1h')
            except Exception as e:
                print(f"\n  [WARN] Fetch error: {e}, retrying in {POLL_INTERVAL}s...")
                time.sleep(POLL_INTERVAL)
                continue

            if latest is None:
                time.sleep(POLL_INTERVAL)
                continue

            new_ts = latest['timestamp']

            if new_ts <= last_candle_ts:
                # No new candle yet -- show waiting status
                now = datetime.now().strftime('%H:%M:%S')
                cur_price = latest['close']
                next_hr = last_candle_ts + pd.Timedelta(hours=1)
                try:
                    mins_left = max(0, (next_hr - pd.Timestamp.now()).total_seconds() / 60)
                except Exception:
                    mins_left = 0
                clear_line()
                print(
                    f'\r  [{now}] Waiting... '
                    f'BTC: ${cur_price:,.2f} | ~{mins_left:.0f}m to next candle',
                    end='', flush=True
                )
                time.sleep(POLL_INTERVAL)
                continue

            # === NEW CANDLE ARRIVED ===
            last_candle_ts = new_ts
            live_step += 1
            print(f"\n\n  [NEW CANDLE] {new_ts} | Close: ${latest['close']:,.2f}")

            # Append new candle to raw data
            new_row = pd.DataFrame([{
                'timestamp': new_ts,
                'open':      latest['open'],
                'high':      latest['high'],
                'low':       latest['low'],
                'close':     latest['close'],
                'volume':    latest['volume'],
            }])
            raw_df = pd.concat([raw_df, new_row], ignore_index=True)

            # Keep rolling window (prevent unbounded growth)
            if len(raw_df) > WARMUP_CANDLES + 100:
                raw_df = raw_df.iloc[-WARMUP_CANDLES:].reset_index(drop=True)

            # Rebuild features on full history
            df = build_features_v3(raw_df.copy())

            # Recreate env with updated data
            env = CryptoTradingEnvV4(df, **env_kwargs)

            # Rebuild vec_env with new env
            model, vec_env = load_agent_v2(env, name='ppo_v4_final')

            # Fast-forward to the last candle
            obs = vec_env.reset()
            n_steps_total = len(df) - SEQUENCE_LENGTH - 1
            for i in range(n_steps_total):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done_ff, info_ff = vec_env.step(action)
                if isinstance(done_ff, np.ndarray):
                    done_ff = done_ff[0]
                if done_ff:
                    break

            if isinstance(info_ff, (list, tuple)):
                info_ff = info_ff[0]

            # Extract the final action and state
            action_int = int(action[0]) if hasattr(action, '__len__') else int(action)
            action_dist[action_int] = action_dist.get(action_int, 0) + 1
            last_action = action_names.get(action_int, '?')

            price    = info_ff.get('current_price', latest['close'])
            balance  = info_ff.get('balance', INITIAL_BALANCE)
            position = info_ff.get('position', 0)
            n_trades = info_ff.get('n_trades', 0)
            total_ret = info_ff.get('total_return', 0.0) * 100

            # Determine active strategy signals
            last_idx = len(df) - 1
            signals = []
            if 'bbrsi_setup' in df.columns and df['bbrsi_setup'].iloc[last_idx] == 1:
                signals.append('BB-RSI')
            if 'macd_adx_setup' in df.columns and df['macd_adx_setup'].iloc[last_idx] == 1:
                signals.append('MACD-ADX')
            if 'mic_setup' in df.columns and df['mic_setup'].iloc[last_idx] == 1:
                signals.append('MIC')
            last_signal = '+'.join(signals) if signals else 'none'

            # Feed action into PaperTrader
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

            if trader.symbol in trader.open_positions:
                trader.check_stop_loss(price)

            # Log equity point
            equity_curve.append({
                'step':      live_step,
                'timestamp': new_ts.isoformat(),
                'price':     round(price, 2),
                'balance':   round(balance, 2),
                'position':  position,
                'action':    last_action,
                'signal':    last_signal,
            })

            # Display dashboard
            trader_pnl = trader.metrics.get('total_pnl', 0.0)
            next_candle = (new_ts + pd.Timedelta(hours=1)).strftime('%H:%M')
            print_dashboard(
                step=live_step,
                price=price,
                position=position,
                balance=balance,
                total_return=total_ret,
                n_trades=n_trades,
                action_name=last_action,
                last_signal=last_signal,
                trader_pnl=trader_pnl,
                next_check=next_candle
            )

            # Auto-save every 10 candles
            if live_step % 10 == 0:
                _save_live_results(equity_curve, action_dist, balance,
                                   total_ret, n_trades, live_step)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("  LIVE TRADING STOPPED (Ctrl+C)")
        print("=" * 70)

    # -- Final save -----------------------------------------------------------
    _save_live_results(equity_curve, action_dist, balance,
                       total_ret, n_trades, live_step)

    # PaperTrader summary
    print("\n  --- PaperTrader Execution Summary ---")
    trader.print_summary()
    trader.save_trades()
    trader.save_metrics()

    print("\n  Live paper trading session ended.")


def _save_live_results(equity_curve, action_dist, balance,
                       total_ret, n_trades, live_step):
    """Save live trading results to JSON."""
    LOG_DIR.mkdir(exist_ok=True)
    results = {
        'timestamp':        datetime.now().isoformat(),
        'mode':             'live',
        'model':            'ppo_v4_final',
        'environment':      'CryptoTradingEnvV4',
        'initial_balance':  INITIAL_BALANCE,
        'current_balance':  round(balance, 2),
        'total_return_pct': round(total_ret, 4),
        'n_trades':         n_trades,
        'live_candles':     live_step,
        'action_dist':      {str(k): v for k, v in action_dist.items()},
        'equity_curve':     equity_curve,
    }
    results_path = LOG_DIR / 'live_paper_trade_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  [SAVED] Results -> {results_path}")


if __name__ == '__main__':
    main()
