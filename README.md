---
title: PPO v4 Crypto Trader
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: true
license: mit
short_description: Live BTC/USDT paper trading with BiLSTM + PPO RL model
---

# PPO v4 Live Crypto Paper Trader

A reinforcement learning trading agent that makes **live trading decisions** on BTC/USDT using a trained PPO model with BiLSTM feature extraction.

## Architecture
- **BiLSTM** (64-dim) compresses 60-candle market patterns
- **PPO v4** (Stable-Baselines3) decides: FLAT / LONG / SHORT
- **18 strategy signals** from BB-RSI, MACD-ADX, and MIC confluence strategies
- **88-dimensional observation space**: 64 BiLSTM + 6 account + 18 strategy

## How It Works
1. Connects to Binance Vision API (public, no credentials needed)
2. Fetches 500 recent 1h candles for warm-up
3. Runs the trained PPO model through history
4. Polls for new candles every 30 seconds
5. Makes live trading decisions and displays results

## Dashboard
- **Live Status**: Real-time BTC price, position, balance, return
- **Equity Curve**: Interactive Plotly chart with trade markers
- **Trade Log**: Full history of model decisions
- **Performance**: Action distribution and metrics
