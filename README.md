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

# PPO v4 Live Crypto Paper Trader 🤖📈

A complete end-to-end Reinforcement Learning trading pipeline. This agent makes **live trading decisions** on BTC/USDT using a trained Proximal Policy Optimization (PPO) model with a BiLSTM feature extraction layer.

![Dashboard Preview](https://img.shields.io/badge/UI-Gradio_Dashboard-blue)
![Algorithm](https://img.shields.io/badge/RL-PPO-green)
![Model](https://img.shields.io/badge/Neural_Net-BiLSTM-orange)

---

## 🧠 Architecture Overview

The system processes market data in real-time and decides whether to go **LONG**, **SHORT**, or remain **FLAT**:

1. **BiLSTM Feature Extractor (64-dim):** Compresses the last 60 hours of raw market features into a dense 64-dimensional temporal representation.
2. **Strategy Indicators (18-dim):** Computes explicit trading signals based on Bollinger Bands + RSI (Mean Reversion), MACD + ADX (Trend Following), and MIC (Multi-Indicator Confluence).
3. **Account State (6-dim):** Tracks current balance, position, and unrealized PnL.
4. **PPO v4 Agent (88-dim Input):** A Stable-Baselines3 PPO agent that takes the combined 88-dimensional state vector and outputs trading actions.

---

## 🚀 How to Run the Project (Step-by-Step)

To train your own model from scratch, follow this pipeline:

### 1. Data Fetching & Cleaning
First, gather historical BTC/USDT data from Binance to train the models.
```bash
python main.py
# Or use the built-in fetcher directly
python -m src.data.fetcher
```
*This downloads 1-hour candle data (OHLCV) and saves it to `data/raw/`.*

### 2. Train the BiLSTM Model
The BiLSTM needs to be pre-trained to understand market patterns before we train the RL agent. It is trained via self-supervised/supervised auxiliary tasks (like predicting next-candle returns).
```bash
python train_bilstm.py
```
*The best weights will be saved to `models_saved/bilstm_best.pt`.*

### 3. Train the PPO Agent (v4)
Once the BiLSTM is trained, freeze its weights and train the PPO agent. The agent explores the environment, executes simulated trades, and learns to maximize its portfolio balance.
```bash
python train_v3.py
```
*This will train the RL agent and save the final model to `models_saved/ppo_v4_final.zip` along with its vector normalization stats `vecnorm_ppo_v4_final.pkl`.*

### 4. Deploy the Live Dashboard
To run the live paper-trading bot with a beautiful UI:
```bash
python app.py
```
This script will:
- Connect to the Binance Vision API (public, no credentials needed).
- Fetch the last 500 hours of candles to "warm up" the indicators (like the 200 EMA).
- Fast-forward the PPO model through the warm-up data.
- Start polling Binance every 30 seconds for new closed candles.
- Make real-time trading decisions.
- Host a **Gradio Dashboard** locally at `http://localhost:7860`.

---

## 📊 The Dashboard

The Gradio UI provides real-time insights into what the AI is thinking:

- **Live Status:** Auto-refreshes every 10 seconds. Shows live BTC price, current position, portfolio balance, and the exact strategy signals that triggered the latest action.
- **Equity Curve:** Interactive Plotly chart showing price action overlaid with **LONG** (green) and **SHORT** (red) trade markers.
- **Trade Log:** A detailed tabular history of every trade the model has executed.
- **Performance:** Win-rates, action distributions (pie chart), and overall PnL metrics.

---

## 🌐 Deploying to Hugging Face Spaces

This repository is pre-configured to be deployed directly to Hugging Face Spaces. 
1. Create a Gradio Space on Hugging Face.
2. Push this repository directly to the Space.
3. Hugging Face will automatically read `app.py` and host the live bot 24/7!

*(Note: Hugging Face free-tier spaces go to sleep after 15 minutes of inactivity. When awoken, the bot will automatically fetch missed history from Binance, fast-forward to catch up, and seamlessly resume live trading without losing sync).*
