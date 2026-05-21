#!/usr/bin/env python3
"""
PPO v4 Live Trading Dashboard
----------------------------------------------------------------------
A Gradio dashboard that runs the PPO model live against Binance data
and displays real-time trading decisions, equity curve, and metrics.

Can run locally or deploy to Hugging Face Spaces.

Usage:
    python dashboard.py
"""

import sys
import json
import time
import threading
import traceback
import numpy as np
import pandas as pd
import ccxt
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Global state shared between the trading thread and the Gradio UI
# ---------------------------------------------------------------------------
@dataclass
class LiveState:
    """Thread-safe container for live trading state."""
    status: str = "Initializing..."
    price: float = 0.0
    position: int = 0          # 0=flat, 1=long, -1=short
    balance: float = 10000.0
    initial_balance: float = 10000.0
    total_return: float = 0.0
    n_trades: int = 0
    last_action: str = "FLAT"
    last_signal: str = "--"
    live_candles: int = 0
    warmup_done: bool = False
    error: str = ""
    equity_curve: list = field(default_factory=list)
    trade_log: list = field(default_factory=list)
    action_dist: dict = field(default_factory=lambda: {"FLAT": 0, "LONG": 0, "SHORT": 0})

STATE = LiveState()
LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Binance connection
# ---------------------------------------------------------------------------
def connect_binance():
    """Connect to Binance (Vision API first, then standard, then US, then KuCoin)."""
    # 1. Vision API (geo-unrestricted, but sometimes still blocked)
    try:
        ex = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
        ex.urls['api']['public'] = 'https://data-api.binance.vision/api/v3'
        t = ex.fetch_ticker('BTC/USDT')
        return ex, t['last']
    except Exception:
        pass
        
    # 2. Standard API
    try:
        ex = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
        t = ex.fetch_ticker('BTC/USDT')
        return ex, t['last']
    except Exception:
        pass
        
    # 3. Binance US (Works in US regions like Hugging Face AWS)
    try:
        ex = ccxt.binanceus({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
        t = ex.fetch_ticker('BTC/USDT')
        return ex, t['last']
    except Exception:
        pass
        
    # 4. KuCoin (Universal fallback for BTC/USDT if Binance is totally blocked)
    try:
        ex = ccxt.kucoin({'enableRateLimit': True})
        t = ex.fetch_ticker('BTC/USDT')
        return ex, t['last']
    except Exception as e:
        return None, 0.0


def fetch_candles(exchange, symbol='BTC/USDT', timeframe='1h', limit=500):
    """Fetch historical candles."""
    all_c = []
    since = None
    remaining = limit
    while remaining > 0:
        batch = min(remaining, 1000)
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=batch)
        if not candles:
            break
        all_c.extend(candles)
        since = candles[-1][0] + 1
        remaining -= len(candles)
        if len(candles) < batch:
            break
        time.sleep(0.2)
    if len(all_c) > limit:
        all_c = all_c[-limit:]
    df = pd.DataFrame(all_c, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df


# ---------------------------------------------------------------------------
# Live trading thread
# ---------------------------------------------------------------------------
def trading_loop():
    """Background thread that runs the PPO model live."""
    global STATE
    try:
        with LOCK:
            STATE.status = "Connecting to Binance..."

        exchange, price = connect_binance()
        if exchange is None:
            with LOCK:
                STATE.status = "OFFLINE - Cannot connect to Binance"
                STATE.error = "Failed to connect to Binance API"
            return

        with LOCK:
            STATE.price = price
            STATE.status = "Loading model..."

        # Import ML modules (heavy imports)
        from config.settings import SEQUENCE_LENGTH, STATE_DIM, MODELS_DIR, SYMBOL
        from src.models.bilstm import load_model as load_bilstm, freeze_model
        from src.data.features_v3 import build_features_v3, BILSTM_FEATURES, PPO_STRATEGY_FEATURES_V4
        from src.environment.trading_env_v4 import CryptoTradingEnvV4
        from src.models.ppo_agent_v2 import load_agent_v2

        # Check model files
        model_path = MODELS_DIR / 'ppo_v4_final.zip'
        vecnorm_path = MODELS_DIR / 'vecnorm_ppo_v4_final.pkl'
        if not model_path.exists() or not vecnorm_path.exists():
            with LOCK:
                STATE.status = "ERROR - Model files not found"
                STATE.error = f"Missing: {model_path.name} or {vecnorm_path.name}"
            return

        # Load BiLSTM
        try:
            bilstm = load_bilstm('bilstm_best.pt')
            freeze_model(bilstm)
            use_lstm = True
        except FileNotFoundError:
            bilstm = None
            use_lstm = False

        with LOCK:
            STATE.status = "Fetching warm-up candles..."

        # Fetch warm-up data
        raw_df = fetch_candles(exchange, SYMBOL, '1h', 500)
        raw_df = raw_df.reset_index()
        raw_df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        last_candle_ts = raw_df['timestamp'].iloc[-1]

        with LOCK:
            STATE.status = "Building features..."

        df = build_features_v3(raw_df.copy())
        feat_cols = [c for c in BILSTM_FEATURES if c in df.columns]

        env_kwargs = dict(
            feature_cols=feat_cols,
            initial_balance=10_000.0,
            transaction_fee=0.001,
            use_lstm_features=use_lstm,
            lstm_model=bilstm,
        )
        env = CryptoTradingEnvV4(df, **env_kwargs)

        with LOCK:
            STATE.status = "Loading PPO model..."

        model, vec_env = load_agent_v2(env, name='ppo_v4_final')

        with LOCK:
            STATE.status = "Warming up model (processing history)..."

        # Fast-forward warm-up
        obs = vec_env.reset()
        warmup_steps = len(df) - SEQUENCE_LENGTH - 2
        action_names = {0: 'FLAT', 1: 'LONG', 2: 'SHORT'}

        for i in range(warmup_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            if isinstance(done, np.ndarray):
                done = done[0]
            if done:
                break
            if (i + 1) % 50 == 0:
                pct = (i + 1) / warmup_steps * 100
                with LOCK:
                    STATE.status = f"Warming up... {pct:.0f}%"

        if isinstance(info, (list, tuple)):
            info = info[0]

        with LOCK:
            STATE.warmup_done = True
            STATE.balance = info.get('balance', 10_000.0)
            STATE.n_trades = info.get('n_trades', 0)
            STATE.position = info.get('position', 0)
            STATE.total_return = info.get('total_return', 0.0) * 100
            STATE.price = float(df['close'].iloc[-1])
            STATE.status = "LIVE - Waiting for next candle..."

        # ---- LIVE LOOP ----
        while True:
            try:
                candles = exchange.fetch_ohlcv(SYMBOL, '1h', limit=2)
                if len(candles) < 2:
                    time.sleep(30)
                    continue
                last_closed = candles[-2]
                new_ts = pd.Timestamp(last_closed[0], unit='ms')
                cur_price = candles[-1][4]  # current candle close

                with LOCK:
                    STATE.price = cur_price

                if new_ts <= last_candle_ts:
                    next_hr = last_candle_ts + pd.Timedelta(hours=1)
                    try:
                        mins = max(0, (next_hr - pd.Timestamp.now()).total_seconds() / 60)
                    except Exception:
                        mins = 0
                    with LOCK:
                        STATE.status = f"LIVE - Next candle in ~{mins:.0f}m"
                    time.sleep(30)
                    continue

                # NEW CANDLE
                last_candle_ts = new_ts
                with LOCK:
                    STATE.live_candles += 1
                    STATE.status = "Processing new candle..."

                new_row = pd.DataFrame([{
                    'timestamp': new_ts,
                    'open': last_closed[1], 'high': last_closed[2],
                    'low': last_closed[3], 'close': last_closed[4],
                    'volume': last_closed[5],
                }])
                raw_df = pd.concat([raw_df, new_row], ignore_index=True)
                if len(raw_df) > 600:
                    raw_df = raw_df.iloc[-500:].reset_index(drop=True)

                df = build_features_v3(raw_df.copy())
                env = CryptoTradingEnvV4(df, **env_kwargs)
                model, vec_env = load_agent_v2(env, name='ppo_v4_final')

                obs = vec_env.reset()
                n_ff = len(df) - SEQUENCE_LENGTH - 1
                for i in range(n_ff):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done_ff, info_ff = vec_env.step(action)
                    if isinstance(done_ff, np.ndarray):
                        done_ff = done_ff[0]
                    if done_ff:
                        break

                if isinstance(info_ff, (list, tuple)):
                    info_ff = info_ff[0]

                action_int = int(action[0]) if hasattr(action, '__len__') else int(action)
                act_name = action_names.get(action_int, '?')

                price = info_ff.get('current_price', last_closed[4])
                balance = info_ff.get('balance', 10_000.0)
                position = info_ff.get('position', 0)
                n_trades = info_ff.get('n_trades', 0)
                total_ret = info_ff.get('total_return', 0.0) * 100

                # Detect signals
                last_idx = len(df) - 1
                signals = []
                for sig in ['bbrsi_setup', 'macd_adx_setup', 'mic_setup']:
                    if sig in df.columns and df[sig].iloc[last_idx] == 1:
                        signals.append(sig.replace('_setup', '').upper())
                sig_str = '+'.join(signals) if signals else 'none'

                with LOCK:
                    STATE.price = price
                    STATE.balance = balance
                    STATE.position = position
                    STATE.n_trades = n_trades
                    STATE.total_return = total_ret
                    STATE.last_action = act_name
                    STATE.last_signal = sig_str
                    STATE.action_dist[act_name] = STATE.action_dist.get(act_name, 0) + 1
                    STATE.equity_curve.append({
                        'time': new_ts.strftime('%m/%d %H:%M'),
                        'price': round(price, 2),
                        'balance': round(balance, 2),
                        'position': position,
                        'action': act_name,
                        'signal': sig_str,
                    })
                    STATE.status = f"LIVE - Last update: {new_ts.strftime('%H:%M')}"

            except Exception as e:
                with LOCK:
                    STATE.error = str(e)
                    STATE.status = f"LIVE - Error (retrying): {str(e)[:50]}"
                time.sleep(30)

    except Exception as e:
        with LOCK:
            STATE.status = f"CRASHED: {str(e)[:80]}"
            STATE.error = traceback.format_exc()


# ---------------------------------------------------------------------------
# Load saved results for initial display
# ---------------------------------------------------------------------------
def load_saved_results():
    """Load backtest or live results for initial display."""
    for fname in ['live_paper_trade_results.json', 'paper_trade_results.json']:
        path = LOG_DIR / fname
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Dashboard UI builders
# ---------------------------------------------------------------------------
DARK_BG = "#0d1117"
CARD_BG = "#161b22"
GREEN = "#3fb950"
RED = "#f85149"
BLUE = "#58a6ff"
YELLOW = "#d29922"
GRAY = "#8b949e"
WHITE = "#e6edf3"

CSS = """
.dashboard-container { font-family: 'Inter', 'Segoe UI', sans-serif; }
.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    min-height: 100px;
}
.metric-value { font-size: 28px; font-weight: 700; margin: 8px 0; }
.metric-label { font-size: 13px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
.status-live { color: #3fb950; font-weight: 600; }
.status-error { color: #f85149; font-weight: 600; }
.pos-flat { color: #8b949e; }
.pos-long { color: #3fb950; }
.pos-short { color: #f85149; }
footer { display: none !important; }
"""

THEME = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)


def make_status_html():
    """Build the live status panel."""
    with LOCK:
        s = STATE

    pos_map = {0: ('FLAT', '#8b949e', '--'), 1: ('LONG', GREEN, '+'), -1: ('SHORT', RED, '-')}
    pos_name, pos_color, pos_icon = pos_map.get(s.position, ('?', GRAY, ''))
    ret_color = GREEN if s.total_return >= 0 else RED
    ret_arrow = "▲" if s.total_return >= 0 else "▼"
    status_color = GREEN if 'LIVE' in s.status else (RED if 'ERROR' in s.status or 'CRASH' in s.status else YELLOW)

    html = f"""
    <div style="font-family: 'Inter', 'Segoe UI', sans-serif; color: {WHITE};">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px; padding: 12px 20px;
                    background: linear-gradient(90deg, {CARD_BG}, #1c2333); border-radius: 10px; border: 1px solid #30363d;">
            <div style="width: 10px; height: 10px; border-radius: 50%; background: {status_color};
                        box-shadow: 0 0 8px {status_color}; animation: pulse 2s infinite;"></div>
            <span style="font-size: 14px; color: {status_color};">{s.status}</span>
            <span style="margin-left: auto; font-size: 12px; color: {GRAY};">Live Candles: {s.live_candles}</span>
        </div>
        <style>@keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}</style>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px;">
            <div style="background: linear-gradient(135deg, #161b22, #1c2333); border: 1px solid #30363d;
                        border-radius: 12px; padding: 20px; text-align: center;">
                <div style="font-size: 12px; color: {GRAY}; text-transform: uppercase; letter-spacing: 1px;">BTC/USDT</div>
                <div style="font-size: 30px; font-weight: 700; color: {WHITE}; margin: 8px 0;">${s.price:,.2f}</div>
            </div>
            <div style="background: linear-gradient(135deg, #161b22, #1c2333); border: 1px solid #30363d;
                        border-radius: 12px; padding: 20px; text-align: center;">
                <div style="font-size: 12px; color: {GRAY}; text-transform: uppercase; letter-spacing: 1px;">Position</div>
                <div style="font-size: 30px; font-weight: 700; color: {pos_color}; margin: 8px 0;">{pos_icon} {pos_name}</div>
            </div>
            <div style="background: linear-gradient(135deg, #161b22, #1c2333); border: 1px solid #30363d;
                        border-radius: 12px; padding: 20px; text-align: center;">
                <div style="font-size: 12px; color: {GRAY}; text-transform: uppercase; letter-spacing: 1px;">Balance</div>
                <div style="font-size: 30px; font-weight: 700; color: {WHITE}; margin: 8px 0;">${s.balance:,.2f}</div>
            </div>
            <div style="background: linear-gradient(135deg, #161b22, #1c2333); border: 1px solid #30363d;
                        border-radius: 12px; padding: 20px; text-align: center;">
                <div style="font-size: 12px; color: {GRAY}; text-transform: uppercase; letter-spacing: 1px;">Return</div>
                <div style="font-size: 30px; font-weight: 700; color: {ret_color}; margin: 8px 0;">
                    {ret_arrow} {s.total_return:+.2f}%</div>
            </div>
            <div style="background: linear-gradient(135deg, #161b22, #1c2333); border: 1px solid #30363d;
                        border-radius: 12px; padding: 20px; text-align: center;">
                <div style="font-size: 12px; color: {GRAY}; text-transform: uppercase; letter-spacing: 1px;">Last Action</div>
                <div style="font-size: 24px; font-weight: 600; color: {BLUE}; margin: 8px 0;">{s.last_action}</div>
                <div style="font-size: 11px; color: {GRAY};">Signal: {s.last_signal}</div>
            </div>
            <div style="background: linear-gradient(135deg, #161b22, #1c2333); border: 1px solid #30363d;
                        border-radius: 12px; padding: 20px; text-align: center;">
                <div style="font-size: 12px; color: {GRAY}; text-transform: uppercase; letter-spacing: 1px;">Trades</div>
                <div style="font-size: 30px; font-weight: 700; color: {WHITE}; margin: 8px 0;">{s.n_trades}</div>
            </div>
        </div>
    </div>
    """
    return html


def make_equity_chart():
    """Build equity curve chart from live data or saved results."""
    with LOCK:
        live_eq = list(STATE.equity_curve)

    # If no live data yet, use saved results
    saved = load_saved_results()
    if not live_eq and saved and 'equity_curve' in saved:
        eq_data = saved['equity_curve']
        steps = [e['step'] for e in eq_data]
        prices = [e['price'] for e in eq_data]
        balances = [e['balance'] for e in eq_data]
        actions = [e.get('action', '') for e in eq_data]
        source = "Backtest Results"
    elif live_eq:
        steps = list(range(len(live_eq)))
        prices = [e['price'] for e in live_eq]
        balances = [e['balance'] for e in live_eq]
        actions = [e.get('action', '') for e in live_eq]
        source = "Live Trading"
    else:
        # Empty chart
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark', paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            title="Waiting for data...",
            font=dict(family="Inter, sans-serif", color=WHITE),
        )
        return fig

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=[f'BTC/USDT Price ({source})', 'Portfolio Balance']
    )

    # Price line
    fig.add_trace(go.Scatter(
        x=steps, y=prices, mode='lines', name='BTC Price',
        line=dict(color=BLUE, width=2),
        fill='tozeroy', fillcolor='rgba(88,166,255,0.05)',
    ), row=1, col=1)

    # Balance line
    fig.add_trace(go.Scatter(
        x=steps, y=balances, mode='lines', name='Balance',
        line=dict(color=GREEN, width=2.5),
        fill='tozeroy', fillcolor='rgba(63,185,80,0.08)',
    ), row=2, col=1)

    # Mark position changes
    for i, act in enumerate(actions):
        if act == 'LONG':
            fig.add_trace(go.Scatter(
                x=[steps[i]], y=[prices[i]], mode='markers',
                marker=dict(color=GREEN, size=8, symbol='triangle-up'),
                showlegend=False, hovertext='LONG',
            ), row=1, col=1)
        elif act == 'SHORT':
            fig.add_trace(go.Scatter(
                x=[steps[i]], y=[prices[i]], mode='markers',
                marker=dict(color=RED, size=8, symbol='triangle-down'),
                showlegend=False, hovertext='SHORT',
            ), row=1, col=1)

    # Initial balance reference line
    fig.add_hline(y=10000, line_dash="dot", line_color=GRAY, opacity=0.5, row=2, col=1)

    fig.update_layout(
        template='plotly_dark', paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(family="Inter, sans-serif", color=WHITE, size=12),
        height=550, margin=dict(l=60, r=30, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
    )
    fig.update_xaxes(gridcolor='#21262d', zeroline=False)
    fig.update_yaxes(gridcolor='#21262d', zeroline=False)

    return fig


def make_trades_table():
    """Build trade history table."""
    with LOCK:
        live_eq = list(STATE.equity_curve)

    if live_eq:
        df = pd.DataFrame(live_eq)
        return df

    saved = load_saved_results()
    if saved and 'equity_curve' in saved:
        eq = saved['equity_curve']
        # Sample every 50 steps for readability
        sampled = eq[::50] + [eq[-1]] if len(eq) > 50 else eq
        df = pd.DataFrame(sampled)
        return df

    return pd.DataFrame(columns=['time', 'price', 'balance', 'action', 'signal'])


def make_performance_chart():
    """Build performance metrics chart."""
    with LOCK:
        ad = dict(STATE.action_dist)

    saved = load_saved_results()
    if not any(ad.values()) and saved and 'action_dist' in saved:
        ad = {{'0': 'FLAT', '1': 'LONG', '2': 'SHORT'}.get(k, k): v
              for k, v in saved['action_dist'].items()}

    if not any(ad.values()):
        ad = {'FLAT': 1, 'LONG': 0, 'SHORT': 0}

    colors = [GRAY, GREEN, RED]
    labels = list(ad.keys())
    values = list(ad.values())

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]],
                        subplot_titles=['Action Distribution', 'Performance Metrics'])

    fig.add_trace(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors),
        textinfo='label+percent', textfont_size=13,
        hole=0.45,
    ), row=1, col=1)

    # Performance bar chart
    with LOCK:
        total_ret = STATE.total_return
        balance = STATE.balance

    if total_ret == 0 and saved:
        total_ret = saved.get('total_return_pct', 0)
        balance = saved.get('final_balance', 10000)

    metrics = ['Return %', 'Balance', 'Trades']
    vals = [total_ret, balance - 10000, STATE.n_trades or (saved or {}).get('n_trades', 0)]
    bar_colors = [GREEN if v >= 0 else RED for v in vals]

    fig.add_trace(go.Bar(
        x=metrics, y=vals,
        marker_color=bar_colors,
        text=[f'{v:+.2f}' if isinstance(v, float) else str(v) for v in vals],
        textposition='outside', textfont_size=13,
    ), row=1, col=2)

    fig.update_layout(
        template='plotly_dark', paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(family="Inter, sans-serif", color=WHITE, size=12),
        height=400, margin=dict(l=40, r=40, t=50, b=30),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor='#21262d')
    fig.update_yaxes(gridcolor='#21262d')

    return fig


# ---------------------------------------------------------------------------
# Gradio App
# ---------------------------------------------------------------------------
def build_app():
    with gr.Blocks(title="PPO v4 Crypto Trader", theme=THEME, css=CSS) as app:
        gr.Markdown("""
        # <center>PPO v4 Live Crypto Trader</center>
        <center style="color: #8b949e; font-size: 14px;">
            BiLSTM + PPO Reinforcement Learning | BTC/USDT 1H | Binance Vision API
        </center>
        """)

        # Refresh button at the top
        refresh_btn = gr.Button(
            "Refresh Dashboard",
            variant="primary",
            size="lg",
        )

        # --- Section 1: Live Status ---
        gr.Markdown("### Live Status")
        status_html = gr.HTML(value=make_status_html)

        # --- Section 2: Equity Curve ---
        gr.Markdown("### Equity Curve")
        equity_plot = gr.Plot(value=make_equity_chart)

        # --- Section 3: Trade Log ---
        gr.Markdown("### Trade Log")
        trades_df = gr.Dataframe(value=make_trades_table,
                                 label="Trade History",
                                 wrap=True)

        # --- Section 4: Performance ---
        gr.Markdown("### Performance")
        perf_plot = gr.Plot(value=make_performance_chart)

        # Wire refresh button to all outputs
        refresh_btn.click(
            fn=lambda: (make_status_html(), make_equity_chart(),
                        make_trades_table(), make_performance_chart()),
            outputs=[status_html, equity_plot, trades_df, perf_plot],
        )

        # Auto-refresh just the status HTML every 10s (lightweight, no tab conflict)
        timer = gr.Timer(value=10)
        timer.tick(fn=make_status_html, outputs=status_html)

        gr.Markdown("""
        <center style="color: #484f58; font-size: 11px; margin-top: 20px;">
            PPO v4 Paper Trader | Model: ppo_v4_final | Env: CryptoTradingEnvV4 (88-dim) |
            Click "Refresh Dashboard" to update charts
        </center>
        """)

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Start trading in background thread
    t = threading.Thread(target=trading_loop, daemon=True)
    t.start()

    # Launch Gradio
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


