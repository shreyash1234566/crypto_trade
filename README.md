# Bi-LSTM + PPO Crypto Trading Bot

A quantitative trading system using Bi-directional LSTM for feature extraction and Proximal Policy Optimization (PPO) for decision making, with Purged Walk-Forward validation following Dr. Marcos Lopez de Prado's methodology.

## Features

- **Bi-LSTM Feature Extractor**: Extracts 64-dimensional market state from price sequences
- **PPO Reinforcement Learning**: 3-action trading (Buy/Hold/Sell) with asymmetric rewards
- **Purged Walk-Forward Validation**: Prevents data leakage in time-series cross-validation
- **Risk Management**: 2% fixed risk, volatility filtering, automatic stop-loss
- **Paper Trading**: Simulated trading on Binance Testnet

## Project Structure

```
crypto_trade/
├── config/
│   └── settings.py            # Configuration and hyperparameters
├── src/
│   ├── data/
│   │   ├── fetcher.py         # CCXT data collection (1-min)
│   │   ├── resampler.py       # Timeframe aggregation
│   │   ├── features.py        # Technical indicators
│   │   └── fear_greed.py      # Sentiment API (optional)
│   ├── models/
│   │   ├── bilstm.py          # Bi-LSTM feature extractor
│   │   └── ppo_agent.py       # PPO agent with custom policy
│   ├── environment/
│   │   └── trading_env.py     # Gymnasium trading environment
│   ├── validation/
│   │   └── purged_cv.py       # Purged Walk-Forward CV
│   └── execution/
│       ├── risk_manager.py    # Position sizing and filters
│       └── paper_trader.py    # Paper trading execution
├── data/
│   ├── raw/                   # 1-min OHLCV data
│   └── processed/             # Processed features
├── requirements.txt
└── main.py                    # CLI orchestrator
```

## Installation

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
copy .env.example .env
# Edit .env with your Binance API keys (Testnet for paper trading)
```

## Quick Start

### 1. Fetch Data (6 months of 1-minute BTC/USDT)
```powershell
python main.py fetch --days 180
```

### 2. Process Data (Resample to 15-min + Features)
```powershell
python main.py process --fear-greed
```

### 3. Pre-train Bi-LSTM
```powershell
python main.py pretrain --epochs 50
```

### 4. Train PPO Agent (with Purged Walk-Forward)
```powershell
python main.py train --timesteps 100000 --use-lstm
```

### 5. Paper Trade
```powershell
python main.py paper --use-lstm --testnet
```

## Key Concepts

### Purged Walk-Forward Validation

Standard cross-validation fails for time-series because it allows future data to leak into training. Our implementation:

1. **Train**: Use past 30 days of data
2. **Purge**: Delete 1 hour of data (4 candles) at the boundary
3. **Test**: Evaluate on next 7 days
4. **Walk Forward**: Slide window and repeat

```
Fold 1: Train [Day 1-30] → Purge [1hr] → Test [Day 31-37]
Fold 2: Train [Day 8-37] → Purge [1hr] → Test [Day 38-44]
...
```

### Reward Function

| Outcome | Reward | Rationale |
|---------|--------|-----------|
| Profit | +1.0 × PnL | Encourage winning trades |
| Loss | -1.5 × PnL | Penalize losses harder (asymmetric) |
| Holding loser | -0.01/step | Force quick exits on bad trades |

### Risk Management

- **Position Sizing**: 2% max risk per trade
- **Volatility Filter**: Reject trades when volatility > 2× average
- **Stop-Loss**: Automatic 3% stop-loss
- **Max Positions**: 1 concurrent position

## Configuration

Edit `config/settings.py` to customize:

```python
# Trading
SYMBOL = "BTC/USDT"
TIMEFRAME_TRADE = "15m"

# Model
SEQUENCE_LENGTH = 60      # 60 candles lookback
STATE_DIM = 64            # Bi-LSTM output dimension

# Risk
MAX_RISK_PER_TRADE = 0.02 # 2%
STOP_LOSS_PCT = 0.03      # 3%
```

## Dependencies

- **mlfinpy**: Purged Walk-Forward (Lopez de Prado methods) - FREE
- **stable-baselines3**: PPO implementation
- **torch**: Bi-LSTM neural network
- **ccxt**: Exchange connectivity
- **gymnasium**: RL environment
- **ta**: Technical indicators

## License

MIT License

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*
- mlfinpy: https://github.com/baobach/mlfinpy
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
